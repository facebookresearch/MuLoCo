# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Optional, TYPE_CHECKING, Union
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.tensor import DTensor
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.components.quantized_outer_sgd import QuantizedOuterSGD
from typing import Callable


from torchtitan.experiments.dion_optimizer.muon import Muon
from torchtitan.experiments.dion_optimizer.muon import MuonOptimizerConfig
from torchtitan.experiments.dion_optimizer.parameter_classification import create_parameter_groups

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    if TYPE_CHECKING:
        from torchft import local_sgd

    has_torchft = True
else:
    has_torchft = False


class FTManager:
    def __init__(
        self,
        manager: Optional["ft.Manager"],
        group_size: int = 1,
        replica_id: int = 0,
    ) -> None:
        self._manager = manager
        self.group_size = group_size
        self.replica_id = replica_id
        if has_torchft and manager is not None:
            self.replicate_pg = ft.process_group.ManagedProcessGroup(self._manager)
            self.replicate_pg.register("dp_replicate")

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def manager(self) -> "ft.Manager":
        assert self._manager is not None
        return self._manager

    def get_dp_info(self, dp_degree: int, dp_rank: int) -> tuple[int, int]:
        return dp_degree * self.group_size, dp_degree * self.replica_id + dp_rank

    def set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
        def all_reduce_hook(output):
            dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)

        def apply_set_all_reduce_hook(m):
            if isinstance(m, FSDPModule):
                m.set_all_reduce_hook(all_reduce_hook)

        for part in model_parts:
            part.apply(apply_set_all_reduce_hook)


def init_ft_manager(job: JobConfig) -> FTManager:
    """Initialize the FT manager if TorchFT is enabled.

    Args:
        job (JobConfig): The job configuration.

    Returns:
        FTManager: A wrapper around TorchFT.Manager
    """
    if not job.fault_tolerance.enable:
        return FTManager(None)

    if not has_torchft:
        raise ImportError("torchft is not installed. Please install it.")

    if job.fault_tolerance.min_replica_size < 1:
        raise ValueError("At least one FT replica is required.")

    pg = ft.ProcessGroupNCCL(timeout=timedelta(seconds=360))

    # If the training method is specific, then the quorum should be synchronous
    use_async_quorum = job.fault_tolerance.semi_sync_method is None

    return FTManager(
        ft.Manager(
            pg=pg,
            min_replica_size=job.fault_tolerance.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            quorum_timeout=timedelta(seconds=720),
            connect_timeout=timedelta(seconds=120),
            timeout=timedelta(seconds=120),
            use_async_quorum=use_async_quorum,
            replica_id=f"torchtitan_ft_{job.fault_tolerance.replica_id}",
        ),
        group_size=job.fault_tolerance.group_size,
        replica_id=job.fault_tolerance.replica_id,
    )


@dataclass
class FTParallelDims(ParallelDims):
    ft_manager: FTManager

    def build_mesh(self, device_type: str) -> DeviceMesh:
        def func(
            device_type: str, mesh_shape: list[int], mesh_dim_names: list[str]
        ) -> DeviceMesh:
            from torchft.process_group import ft_init_device_mesh

            return ft_init_device_mesh(
                device_type=device_type,
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names,
                replicate_dim=mesh_dim_names.index("dp_replicate"),
                manager=self.ft_manager.manager,
            )

        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1 or name == "dp_replicate":
                dims.append(d)
                names.append(name)

        return self._build_mesh(device_type, dims, names, func)

    @property
    def dp_replicate_enabled(self):
        return True


def ft_dist_reduce(
    x: torch.Tensor, reduceOp: str, mesh: DeviceMesh
) -> tuple[torch.Tensor, str, DeviceMesh]:
    if has_torchft and isinstance(mesh, ft.device_mesh._FlattenDeviceMesh):
        x = funcol.all_reduce(
            x, reduceOp=reduceOp, group=mesh.managed_mesh.replicate_pg
        )
        return x, reduceOp, mesh.managed_mesh.mesh
    return x, reduceOp, mesh


def ft_clip_grad_norm_util(total_norm: DTensor) -> torch.Tensor:
    if has_torchft:
        mesh = total_norm._spec.mesh
        if isinstance(mesh, ft.device_mesh.ManagedDeviceMesh):
            # The gradients along the replicated dim has already been reduced.
            # So we don't need another reducution beforing removing the
            # replicate dimension
            local_tensor = total_norm.to_local()
            placements = list(copy.copy(total_norm._spec.placements))
            placements.pop(mesh.replicate_dim)
            return DTensor.from_local(local_tensor, mesh.mesh, placements)

    return total_norm


def maybe_semi_sync_training(
    config: JobConfig,
    ft_manager: FTManager,
    model_parts: list[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fragment_fn: Optional[Callable[..., list[torch.nn.Module]]] = None,
) -> ContextManager[Union["local_sgd.DiLoCo", "local_sgd.LocalSGD", None]]:
    """
    If TorchFT is enabled and the config is set, use semi_sync_method
    """
    ft_config = config.fault_tolerance
    semi_sync_method = ft_config.semi_sync_method
    torchft_enabled = ft_config.enable
    if torchft_enabled and semi_sync_method is not None:
        from torchft import local_sgd

        assert (
            ft_manager._manager is not None
        ), "FTManager must be enabled to use semi-sync training."
        if semi_sync_method.lower() == "diloco":
            if fragment_fn:
                assert len(model_parts) == 1
                model_parts = fragment_fn(model_parts[0], ft_config, config.model.full_model_args['n_layers'])


   
            # model_args.n_layers
            # Create the outer optimizer based on the inner optimizer parameters.
            # params = [group["params"] for group in optimizer.param_groups]
            # params = [param for sublist in params for param in sublist]
            
            optimizers = {
                'sgd': torch.optim.SGD,
                'adamw': torch.optim.AdamW,
                'adam': torch.optim.Adam,
                'adamax': torch.optim.Adamax,
                'quantized_outer_sgd': QuantizedOuterSGD,
                'muon': Muon
            }
            # TODO: fix the outer optimizer to be a list of optimizers
            # outer_optimizer = optimizers[config.outer_optimizer.class_name](
            #     params, **config.outer_optimizer.kwargs, 
            # )


            outer_optimizers = []
            for model in model_parts:

                if config.outer_optimizer.class_name == "muon":
                    muon_config = MuonOptimizerConfig(
                        lr=config.outer_optimizer.kwargs.get("lr", 0.01),
                        mu=config.outer_optimizer.kwargs.get("mu", 0.95),
                        betas=config.outer_optimizer.kwargs.get("betas", (0.9, 0.95)),
                        weight_decay=config.outer_optimizer.kwargs.get("weight_decay", 0.01),
                        epsilon=config.outer_optimizer.kwargs.get("epsilon", 1e-8),
                        nesterov=config.outer_optimizer.kwargs.get("nesterov", False),
                        adjust_lr=config.outer_optimizer.kwargs.get("adjust_lr", "spectral_norm"),
                        flatten=config.outer_optimizer.kwargs.get("flatten", False),
                        use_triton=config.outer_optimizer.kwargs.get("use_triton", False),
                    )
                    params = create_parameter_groups([model], muon_config)
                else:
                    params = [p for p in model.parameters() if p.requires_grad]

                if config.outer_optimizer.class_name == "tree_loco_outer_sgd":
                    config.outer_optimizer.kwargs['worker_id'] = config.global_rank

                outer_optimizer = optimizers[config.outer_optimizer.class_name](
                    params, **config.outer_optimizer.kwargs, 
                )
                # outer_optimizer = torch.optim.SGD(
                #     params, lr=0.7, momentum=0.9, nesterov=True
                # )
                outer_optimizers.append(outer_optimizer)


            if ft_config.backup_device == "cuda":
                backup_device = device
            else:
                backup_device = None
                

            return local_sgd.DiLoCo(
                manager=ft_manager._manager,
                model_fragments=model_parts,
                inner_optimizer=optimizer,
                outer_optimizer=outer_optimizers,
                sync_every=ft_config.sync_steps,
                should_quantize=ft_config.should_quantize,
                fragment_sync_delay=ft_config.fragment_sync_delay,
                fragment_update_alpha=ft_config.fragment_update_alpha,
                backup_device=backup_device,
                use_continuous_centering=ft_config.use_continuous_centering,
                use_periodic_centering=ft_config.use_periodic_centering,
                use_gpa=ft_config.use_gpa,
                use_sparseloco=ft_config.use_sparseloco,
                use_quantized_outer_sgd=ft_config.use_quantized_outer_sgd,
                save_pseudograds=ft_config.save_pseudograds,
                pseudograd_path=ft_config.pseudograd_path,
                global_rank=config.global_rank,
                fragment_kwargs=ft_config.fragment_kwargs,
            )
        elif semi_sync_method.lower() == "local_sgd":
            assert len(model_parts) == 1
            return local_sgd.LocalSGD(
                manager=ft_manager._manager,
                model=model_parts[0],
                optimizer=optimizer,
                sync_every=ft_config.sync_steps,
            )
        else:
            raise ValueError(
                f"Unknown training method: {semi_sync_method}, only 'diloco' and 'local_sgd' are supported."
            )
    return nullcontext()
