# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
import pprint
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional

import torch
from torch.cuda import FloatStorage
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderStopIteration
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
    Timing,
)
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.utils import device_module, device_type
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from mmengine.config import Config, DictAction

from torchtitan.components.fragment import fragment_llm


def get_alternative_dataloader(
    job_config: JobConfig, 
    data_cfg,
    data_type: str="train", 
    dp_world_size: int=1, 
    dp_rank: int=0, 
    gradient_accumulation_steps: int=1):        
    import sys
    import yaml
    from pathlib import Path
    from apps.llm.data_builder import DataSourceArgs, TokenizerArgs
    from apps.llm.args import build_training_iterator
    import pprint
    # pprint.pprint(job_config.to_dict())
    # Build args object to mimic what would be passed to build_training_iterator
    class Args:
        pass
    args = Args()
    
    class Data:
        pass
    data = Data()

    # Sources: use DataSourceArgs to ensure all required fields are present
    if data_type == "train":
        data.sources = [
            DataSourceArgs(
                path=src.path,
                weight=src.weight,
                pattern=src.pattern,
                # type, file_type, and other fields will use DataSourceArgs defaults
            )
            for src in data_cfg.sources
        ]
        args.batch_size = job_config.training.local_batch_size
        data.num_workers = job_config.training.train_num_workers
        data.seed = job_config.training.seed
    elif data_type == "val":
        data.sources = [
            DataSourceArgs(
                path=src.path,
                weight=src.weight,
                pattern=src.pattern,
                # type, file_type, and other fields will use DataSourceArgs defaults
            )
            for src in data_cfg.sources
        ]
        args.batch_size = job_config.training.eval_local_batch_size
        data.num_workers = job_config.training.val_num_workers
        data.seed = 0 # No shuffling for validation
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    # Other data args (set to defaults or from config if present)
    data.shuffle_buffer_size = 1000
    data.shuffle_buffer_allow_flush = False
    data.partition_by_bucket = None
    data.max_precompute = 40
    args.data = data
    args.seq_len = job_config.training.seq_len

    #important for dataloader checkpointing
    # this is used to ensure the dataloader is checkpointed at every step.
    args.checkpoint_freq = 1 #job_config.checkpoint.interval
    args.grad_acc_steps = 1 #gradient_accumulation_steps

    # Tokenizer
    tokenizer_cfg = data_cfg.tokenizer
    data.tokenizer = TokenizerArgs(
        name=tokenizer_cfg.get("name"),
        path=tokenizer_cfg.get("path"),
    )

    dataset_loader = build_training_iterator(args, dp_world_size, dp_rank)

    return dataset_loader






def get_wandb_name(job_config: JobConfig, dp_degree: int) -> str:
    # Determine method name
    if job_config.fault_tolerance.enable:
        # H: local batch size, K: number of workers, B: global batch size
        H = job_config.fault_tolerance.sync_steps
        K = job_config.fault_tolerance.group_size
        if job_config.fault_tolerance.use_periodic_centering:
            prefix = f"diloco-periodic"
        elif job_config.fault_tolerance.use_continuous_centering:
            prefix = f"diloco-continuous"
        elif job_config.fault_tolerance.use_gpa:
            prefix = f"diloco-GPA"
        elif job_config.fault_tolerance.use_sparseloco:
            prefix = f"diloco-sparseloco"
        else:
            prefix = f"diloco"

        method = f"{prefix}_{job_config.optimizer.name}_{job_config.outer_optimizer.class_name}_H{H}_K{K}"
    else:
        method = f"data-parallel_{job_config.optimizer.name}"


    if job_config.training.global_batch_size == -1:
        B = job_config.training.local_batch_size * dp_degree
    else:
        B = job_config.training.global_batch_size

    SL = job_config.training.seq_len
    model_name = job_config.model.flavor

    return f"{method}_B{B}_SL{SL}_{model_name}{job_config.metrics.wandb_suffix}"

def update_pseudograds_path(job_config: JobConfig):
    if job_config.fault_tolerance.save_pseudograds:
        h = job_config.fault_tolerance.sync_steps
        k = job_config.fault_tolerance.group_size
        opt = job_config.optimizer.name
        checkpoint_path = job_config.checkpoint.initial_load_path
        # Extract step number from checkpoint path (e.g., "step-780" -> 780)
        import re
        step_match = re.search(r'step-(\d+)', checkpoint_path) if checkpoint_path else None
        step_number = step_match.group(1) if step_match else "unknown"

        job_config.fault_tolerance.pseudograd_path = os.path.join(
            job_config.fault_tolerance.pseudograd_path, 
            f"k{k}_h{h}_opt{opt}_ckpt-steps{step_number}_gsb{job_config.training.global_batch_size}"
        )
    return job_config

def get_checkpoint_name(job_config: JobConfig, dp_degree: int, parallel_dims: ParallelDims) -> str:
    # Determine method name
    if job_config.fault_tolerance.enable:
        # H: local batch size, K: number of workers, B: global batch size
        H = job_config.fault_tolerance.sync_steps
        K = job_config.fault_tolerance.group_size

        
        suffix = f"_alpha{job_config.fault_tolerance.fragment_update_alpha}"

        if job_config.fault_tolerance.use_periodic_centering:
            prefix = f"DiL-periodic"
        elif job_config.fault_tolerance.use_continuous_centering:
            prefix = f"DiL-continuous"
        elif job_config.fault_tolerance.use_gpa:
            prefix = f"DiL-GPA"
        elif job_config.fault_tolerance.use_sparseloco:
            prefix = f"DiL-sparseloco"
            ookw = job_config.outer_optimizer.kwargs
            suffix = suffix + f"_chunk{ookw.chunk_size}_tk{ookw.top_k}_q{ookw.quantization_bins}_r{ookw.quantization_range}"
        elif job_config.fault_tolerance.use_quantized_outer_sgd and job_config.outer_optimizer.class_name != "tree_loco_outer_sgd":
            prefix = f"DiL-qsgd"
            ookw = job_config.outer_optimizer.kwargs
            suffix = suffix + f"_q{ookw.quantization_bins}_r{ookw.quantization_range}_strat-{ookw.compressor_type}_efb{ookw.error_decay}_skip_emb{ookw.skip_embedding_quantization}_topk{ookw.topk_sparsity_ratio}_use_ef{ookw.use_ef}"
        else:
            prefix = f"DiL"

        if job_config.optimizer.name == "GPA":
            method = f"{prefix}_{job_config.optimizer.name}_LR{job_config.optimizer.gpa_kwargs.lr}_{job_config.outer_optimizer.class_name}_LR{job_config.outer_optimizer.kwargs.lr}_m{job_config.outer_optimizer.kwargs.momentum}_H{H}_K{K}{suffix}"
        elif job_config.optimizer.name == "AdEMAMix":
            method = f"{prefix}_{job_config.optimizer.name}_LR{job_config.optimizer.ademamix_kwargs.lr}_{job_config.outer_optimizer.class_name}_LR{job_config.outer_optimizer.kwargs.lr}_m{job_config.outer_optimizer.kwargs.momentum}_H{H}_K{K}{suffix}"
        else:
            method = f"{prefix}_{job_config.optimizer.name}_LR{job_config.optimizer.lr}_{job_config.outer_optimizer.class_name}_LR{job_config.outer_optimizer.kwargs.lr}_m{job_config.outer_optimizer.kwargs.momentum}_H{H}_K{K}{suffix}"
    else:

        if job_config.optimizer.name == "AdEMAMix":
            beta_string = "-".join(map(str, job_config.optimizer.ademamix_kwargs.betas))
            method = f"DP_{job_config.optimizer.name}_LR{job_config.optimizer.ademamix_kwargs.lr}_betas{beta_string}_wd{round(job_config.optimizer.ademamix_kwargs.weight_decay, 7)}_aw{job_config.optimizer.ademamix_kwargs.alpha_warmup}_bw{job_config.optimizer.ademamix_kwargs.beta3_warmup}"
        else:
            method = f"DP_{job_config.optimizer.name}_LR{job_config.optimizer.lr}_b1{job_config.optimizer.beta1}_b2{job_config.optimizer.beta2}_wd{round(job_config.optimizer.weight_decay, 7)}"





    if job_config.training.global_batch_size == -1:
        B = job_config.training.local_batch_size * dp_degree
    else:
        B = job_config.training.global_batch_size

    SL = job_config.training.seq_len
    model_name = job_config.model.flavor
    

    suffix = f"steps{job_config.training.steps}_dps{parallel_dims.dp_shard}_dpr{parallel_dims.dp_replicate}_cp{parallel_dims.cp}_tp{parallel_dims.tp}_pp{parallel_dims.pp}_B{B}_SL{SL}_{model_name}{job_config.metrics.wandb_suffix}"

    return f"{method}_{suffix}"


class LoaderInputLabelWrapper:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            tokens = batch.val.cuda(non_blocking=True)
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]
            # Optionally handle mask if needed
            # labels_mask = (
            #     batch.mask.cuda(non_blocking=True)[:, 1:]
            #     if batch.mask is not None
            #     else None
            # )
            yield [{'input': input_ids}, labels]

    def __len__(self):
        return len(self.loader)

    def state_dict(self) -> dict[str, Any]:
        return self.loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loader.load_state_dict(state_dict)


        
class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    job_config: JobConfig
    gc_handler: utils.GarbageCollection

    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec
    world_mesh: torch.distributed.DeviceMesh
    gradient_accumulation_steps: int

    dataloader: train_spec_module.BaseDataLoader
    metrics_processor: train_spec_module.MetricsProcessor
    checkpointer: CheckpointManager
    train_context: Generator[None, None, None]

    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer

    pp_has_first_stage: bool
    pp_has_last_stage: bool

    device: torch.device

    # states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config
        if self.job_config.metrics.enable_wandb == False:
            self.job_config.metrics.enable_wandb = True
            os.environ['WANDB_MODE'] = 'offline'



        # self.job_config.job.dump_folder = self.job_config.job.dump_folder + "_" + os.environ['SLURM_JOB_ID']


        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not parallelism_config.disable_loss_parallel,
        )
        dist_utils.init_distributed(job_config)

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)
        parallel_dims.world_mesh = world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.job_config.global_rank = torch.distributed.get_rank()
        if self.job_config.fault_tolerance.enable:
            self.job_config.global_rank += self.job_config.fault_tolerance.replica_id * dp_degree

        if self.job_config.fault_tolerance.group_size > 1:
            self.job_config.global_world_size = self.job_config.fault_tolerance.group_size * dp_degree
        else:
            self.job_config.global_world_size = dp_degree

        logger.info(f"Job config: {self.job_config.global_rank}")
        # exit(0)s

        self.ft_manager = ft.init_ft_manager(job_config)
        # If TorchFT is enabled, the dp_rank and dp_degree, which are used for
        # dataloader must be changed.

        # if self.job_config.fault_tolerance.use_sparseloco:
        #     print("================================================")
        #     print("All gather test!")
        #     print("================================================")

        #     grads = {}
        #     grads['indices'] = torch.ones((10,), device=self.device) * self.job_config.fault_tolerance.replica_id
        #     self.ft_manager._manager._init_sync = False 
        #     self.ft_manager._manager.start_quorum(allow_heal=False)
        #     future = self.ft_manager._manager.allgather(grads['indices'])
        #     grads['indices'] = future
        #     print("Future:", future)
        #     temp = future.wait()
        #     print("grads['indices']", grads['indices'])


        #     print(len(temp))
        #     print(type(temp))
        #     if  self.job_config.fault_tolerance.replica_id == 0:
        #         print("grads", grads)
        #         for x in temp:
        #             print(x)


        #     import time 
        #     time.sleep(10)
        #     # print("Future result:", future.wait())
        #     self.close()
        #     torch.distributed.destroy_process_group()
        #     exit(0)




        if self.ft_manager.enabled:
            orig_dp_degree, orig_dp_rank = dp_degree, dp_rank
            dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)



        


        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0

        if self.ft_manager.enabled:
            assert (
                global_batch_size % (job_config.training.local_batch_size * orig_dp_degree) == 0
            ), (
                f"global batch size must be multiple of local batch size times "
                f"data-parallel degree ({global_batch_size} "
                f"% ({job_config.training.local_batch_size} * {orig_dp_degree}) != 0)"
            )
            assert job_config.training.eval_global_batch_size % (job_config.training.eval_local_batch_size * orig_dp_degree) == 0, (
                f"eval global batch size must be multiple of eval local batch size times "
                f"data-parallel degree ({job_config.training.eval_global_batch_size} "
                f"% ({job_config.training.eval_local_batch_size} * {orig_dp_degree}) != 0)"
            )
        else:
            assert (
                global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
            ), (
                f"global batch size must be multiple of local batch size times "
                f"data-parallel degree ({global_batch_size} "
                f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
            )
            assert job_config.training.eval_global_batch_size % (job_config.training.eval_local_batch_size * dp_degree) == 0, (
                f"eval global batch size must be multiple of eval local batch size times "
                f"data-parallel degree ({job_config.training.eval_global_batch_size} "
                f"% ({job_config.training.eval_local_batch_size} * {dp_degree}) != 0)"
            )


        # calculate gradient accumulation steps before data loading
        if self.ft_manager.enabled:
            self.gradient_accumulation_steps = global_batch_size // (
                job_config.training.local_batch_size * orig_dp_degree
            )
            self.ppl_eval_steps = job_config.training.eval_global_batch_size // (
                job_config.training.eval_local_batch_size * orig_dp_degree
            )
        else:
            self.gradient_accumulation_steps = global_batch_size // (
                job_config.training.local_batch_size * dp_degree
            )
            self.ppl_eval_steps = job_config.training.eval_global_batch_size // (
                job_config.training.eval_local_batch_size * dp_degree
            )




        self.dataloader, self.tokenizer = get_alternative_dataloader(
            job_config, 
            data_cfg=job_config.data_train, 
            data_type="train", 
            dp_world_size=dp_degree, 
            dp_rank=dp_rank, 
            gradient_accumulation_steps=self.gradient_accumulation_steps)
        self.dataloader = LoaderInputLabelWrapper(self.dataloader)

        self.val_dataloaders = {}
        for x in job_config.data_val:
            name = x['name']
            temp_dataloader, _ = get_alternative_dataloader(
                job_config, 
                data_cfg=x, 
                data_type="val", 
                dp_world_size=dp_degree, 
                dp_rank=dp_rank, 
                gradient_accumulation_steps=self.ppl_eval_steps)
            self.val_dataloaders[name] = iter(LoaderInputLabelWrapper(temp_dataloader))





        # build model (using meta init)
        model_cls = self.train_spec.cls
        model_args = self.train_spec.config[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config, self.tokenizer)

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = model_cls(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        print("model_args",model_args.__dict__)
        job_config.model.full_model_args = model_args.__dict__
        print("job_config.model.full_model_args",job_config.model.full_model_args)
        print("job_config.model.full_model_args as dict",job_config.to_dict()['model']['full_model_args'])


        # exit(0)

        print("before build_metrics_processor")

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        job_config.wandb_name = get_wandb_name(job_config, dp_degree)
        job_config.checkpoint_name = get_checkpoint_name(job_config, dp_degree, parallel_dims)
        update_pseudograds_path(job_config)

        ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None

        # if self.ft_manager.enabled:
        #     self.ft_manager.manager.start_quorum(allow_heal=False)
        
        self.metrics_processor = build_metrics_processor_fn(
            job_config, 
            parallel_dims, 
            model_args, 
            mesh=self.world_mesh["dp_cp"] if job_config.parallelism.tensor_parallel_degree == 1 else self.world_mesh["tp"], 
            ft_pg=ft_pg,
            device=self.device
        )
        color = self.metrics_processor.color

        # exit(0)

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(job_config)


        
            
        assert self.ppl_eval_steps > 0
        self.loss_fn_eval = rescale_accumulated_loss(
            self.loss_fn, self.ppl_eval_steps
        )
            
        assert self.gradient_accumulation_steps > 0
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )




        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                world_mesh,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]





        # Print layer weights and their mean values for debugging
        # for i, model_part in enumerate(self.model_parts):
        #     logger.info(f"Model part {i} weights:")
        #     for name, param in model_part.named_parameters():
        #         if param.data is not None:
        #             mean_val = param.data.mean().item()
        #             logger.info(f"  {name}: mean = {mean_val:.6f}, shape = {param.shape}")

        # print("EXITING")
        # exit(0)
        
        if (
            self.ft_manager.enabled
            and job_config.fault_tolerance.semi_sync_method is None
        ):
            self.ft_manager.set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config, self.ft_manager, parallel_dims
        )
        # import time
        # time.sleep(5)
        # print("inside train.py")
        # print(type(self.optimizers.optimizers[0]))
        # print(type(self.optimizers.muon_optimizer))
        # exit(0)
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0

        self.semi_sync = ft.maybe_semi_sync_training(
                self.job_config,
                ft_manager=self.ft_manager,
                model_parts=self.model_parts,
                optimizer=self.optimizers,
                device=self.device,
                fragment_fn=fragment_llm,
            )

        states = {"train_state": self}


        if self.ft_manager.enabled and self.job_config.fault_tolerance.semi_sync_method is not None:
            try:
                outer_optimizers = {f"outer_optimizer_{i}": x._outer_optimizer for i,x in enumerate(self.semi_sync._fragments)}
                states.update(outer_optimizers)
            except Exception as e:
                logger.warning("No outer optimizer found in semi_sync method, skipping checkpointing for the outer optimizer.")

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=states,
            job_config=job_config,
            ft_manager=self.ft_manager,
            delete_checkpoints_at_last_step=job_config.checkpoint.delete_checkpoints_at_last_step,
        )
        

        self.train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})."
        )

        if self.job_config.fault_tolerance.enable:
            # For fault tolerance with diloco, choose steps that are multiples of sync_steps
            sync_steps = self.job_config.fault_tolerance.sync_steps
            total_steps = self.job_config.training.steps
            
            # Calculate approximate 25%, 50%, 75% that are multiples of sync_steps
            step_25 = ((total_steps // 4) // sync_steps) * sync_steps
            step_50 = ((total_steps // 2) // sync_steps) * sync_steps
            step_75 = ((total_steps * 3 // 4) // sync_steps) * sync_steps
            
            # Penultimate step
            penultimate_step = total_steps - 1
            
            self.lm_evaluate_steps = [step_25, step_50, step_75, penultimate_step]
            # Remove duplicates and zero values, keep sorted
            self.lm_evaluate_steps = sorted(list(set([s for s in self.lm_evaluate_steps if s > 0])))
        else:
            # For DDP training, use 25%, 50%, 75%, and penultimate step
            total_steps = self.job_config.training.steps
            step_25 = total_steps // 4
            step_50 = total_steps // 2
            step_75 = total_steps * 3 // 4
            penultimate_step = total_steps - 1
            
            self.lm_evaluate_steps = [step_25, step_50, step_75, penultimate_step]
            # Remove duplicates and zero values, keep sorted
            self.lm_evaluate_steps = sorted(list(set([s for s in self.lm_evaluate_steps if s > 0])))

        
    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderStopIteration() from ex
            input_dict, labels = batch

            # Move tensors to the appropriate device
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            self.metrics_processor.ntokens_since_last_log += labels.numel()
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            yield input_dict, labels


    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["input"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=self.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    
    def eval_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["input"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=self.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline Parallel is not supported for forward only step")
           
        else:
            model_parts[0].eval()
            # Non-PP forward / backward
            with torch.no_grad():
                with self.train_context(optional_context_parallel_ctx):
                    assert len(model_parts) == 1
                    with self.maybe_enable_amp:
                        pred = model_parts[0](inputs)
                        loss = self.loss_fn_eval(pred, labels)
                        del pred
            model_parts[0].train()
        
        return loss


    def eval_step_hellaswag(self) -> dict:
        """
        Run LM Evaluation Harness (hellaswag and other tasks) on the current model.
        
        Uses TorchTitan's mesh-based distributed operations for compatibility with
        various parallelism configurations (DP, FSDP, CP, etc.) and DiLoCo.
        
        Returns:
            Tuple of (results dict, flat_metrics dict for logging)
        """
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline Parallel is not supported for hellaswag evaluation")

        model_parts = self.model_parts

        ############################################################
        # Remove paths from sys so we can import datasets
        ############################################################
        import sys
        original_sys_path = sys.path.copy()
        sys.path = sys.path[5:]
        # must only be imported here so we can remove paths from sys
        from torchtitan.components.lm_eval_helper import evaluate_torchtitan_hellaswag

        
        model_parts[0].eval()
        # Determine ft_pg based on ft_manager status
        # When ft_manager.enabled is True (DiLoCo), we reduce across both dp_cp mesh and ft_pg
        # When ft_manager.enabled is False, we only reduce across dp_cp mesh
        ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None

        # print("device_module.current_device()",device_module.current_device())

        torch.distributed.barrier(device_ids=[device_module.current_device()])
        
        # Must specify device_ids to use GPU/NCCL backend - ft_pg is a TorchFT ManagedProcessGroup
        # that doesn't have a CPU backend, so barrier() without device_ids fails
        if ft_pg is not None:
            torch.distributed.barrier(group=ft_pg, device_ids=[device_module.current_device()])

       
        
        results, flat_metrics = evaluate_torchtitan_hellaswag(
            model=model_parts[0],
            tokenizer=self.tokenizer,
            max_seq_len=self.job_config.training.seq_len,
            max_batch_size=self.job_config.training.eval_local_batch_size,
            global_rank=self.job_config.global_rank,
            global_world_size=self.job_config.global_world_size,
            limit=sys.maxsize,
            num_fewshot=0,
            log_samples=False,
            # TorchTitan-specific parameters for parallelism support
            world_mesh=self.world_mesh,
            parallel_dims=self.parallel_dims,
            train_context=self.train_context,
            maybe_enable_amp=self.maybe_enable_amp,
            job_config=self.job_config,
            ft_pg=ft_pg,
        )

        if self.job_config.global_rank == 0:
            pprint.pprint(flat_metrics)

        model_parts[0].train()

        # reset sys path
        sys.path = original_sys_path

        return results, flat_metrics



    def eval_step_outer(self) -> torch.Tensor:
        # Eval before optimizer step so that training loss is logged on 
        # the same model parameters as evaluation loss
        global_avg_val_loss = global_max_val_loss = None
        if self.step % self.job_config.metrics.eval_freq == 0 or self.step in [0, 1]:
            val_metrics = {}

            for name, dataloder in self.val_dataloaders.items():
                val_losses = []
                for microbatch in range(self.ppl_eval_steps):
                    val_loss = self.eval_step(*next(dataloder))
                    val_losses.append(val_loss.detach())
                val_loss = torch.sum(torch.stack(val_losses))

                if self.parallel_dims.dp_cp_enabled or self.ft_manager.enabled:
                    ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
                    
                    global_avg_val_loss, global_max_val_loss = (
                        dist_utils.dist_mean(val_loss, self.world_mesh["dp_cp"], ft_pg),
                        dist_utils.dist_max(val_loss, self.world_mesh["dp_cp"], ft_pg),
                    )
                else:
                    global_avg_val_loss = global_max_val_loss = val_loss.detach().item()

                val_metrics[f"loss_metrics/global_avg_val_loss/{name}"] = global_avg_val_loss
                val_metrics[f"loss_metrics/global_max_val_loss/{name}"] = global_max_val_loss

            return val_metrics
        else: 
            return None

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):


        # print("inside train_step() self.ft_manager._manager._quorum_future =", self.ft_manager._manager._quorum_future)
        self.optimizers.zero_grad()

        if self.job_config.fault_tolerance.use_continuous_centering and \
            self.job_config.fault_tolerance.semi_sync_method == 'diloco':
            lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
            # print("last lr =", lr)
            self.semi_sync.recenter_parameters(self.job_config.fault_tolerance.fragment_update_alpha * lr)
        elif self.job_config.fault_tolerance.use_periodic_centering and \
            self.job_config.fault_tolerance.semi_sync_method == 'diloco':
            lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
            self.semi_sync._save_lr_to_fragments(lr)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        accumulated_losses = []
        # If data runs out during gradient accumulation, that
        # entire step will not be executed.
        for microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            loss = self.forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss.detach())

        if self.job_config.parallelism.tensor_parallel_degree == 1:
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.job_config.training.max_norm,
                foreach=True,
                pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
            )
        else:
            grad_norm = torch.tensor(0.0)


            
        self.checkpointer.maybe_wait_for_staging()
        # print("Before optimizers.step() inside train_step() self.ft_manager._manager._quorum_future =", self.ft_manager._manager._quorum_future)

        loss = torch.sum(torch.stack(accumulated_losses))
        # self.semi_sync.update_loss_ema(loss)

        with Timing("optimizers.step"):
            self.optimizers.step()
        self.lr_schedulers.step()

        
        global_avg_val_loss = global_max_val_loss = None
        with Timing("eval_step"):
            val_to_log = self.eval_step_outer()

        # Print learning rates for each parameter group
        # for i, optimizer in enumerate(self.optimizers.optimizers): # if hasattr(self.optimizers, 'optimizers') else [self.optimizers]):
        #     for j, param_group in enumerate(optimizer.param_groups):
        #         print(f"[type(optimizer){type(optimizer)}] Optimizer {i}, Param Group {j}: lr = {param_group['lr']}")# algo={param_group['algorithm']}")
        # print("After optimizers.step() inside train_step() self.ft_manager._manager._quorum_future =", self.ft_manager._manager._quorum_future)

        # Reduce the data collected over gradient accumulation steps.
        # loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return


        with Timing("dist_mean_train"):
            if parallel_dims.dp_cp_enabled or self.ft_manager.enabled:
                
                loss = loss.detach()
                # Skip ft manager communication when using semi sync training
                # use_ft_pg = True
                # use_ft_pg = (
                #     self.ft_manager.enabled
                #     and self.job_config.fault_tolerance.semi_sync_method is None
                # )
                 
                if self.ft_manager.enabled and self.job_config.fault_tolerance.group_size > 1:
                    # print("using replicate ft pg")
                    ft_pg = self.ft_manager.replicate_pg
                else:
                    # print("not using replicate ft pg")
                    ft_pg = None
                # Run all 4 reductions in parallel for better performance
                # global_avg_loss, global_max_loss, global_ar_loss, global_ar_max_loss = (
                #     dist_utils.dist_reduce_parallel(
                #         tensors=[loss, loss, loss, loss],
                #         reduceOps=["AVG", "MAX", "AVG", "MAX"],
                #         mesh=self.world_mesh["dp_cp"],
                #         extra_pgs=[None, None, ft_pg, ft_pg],
                #     )
                # )
                    
                global_avg_loss, global_max_loss, global_ar_loss, global_ar_max_loss = (    
                    dist_utils.dist_mean(loss, self.world_mesh["dp_cp"], None),
                    dist_utils.dist_max(loss, self.world_mesh["dp_cp"], None),
                    dist_utils.dist_mean(loss, self.world_mesh["dp_cp"], ft_pg),
                    dist_utils.dist_max(loss, self.world_mesh["dp_cp"], ft_pg),
                )
            else:
                global_avg_loss = global_max_loss = loss.detach().item()






                

        

        if self.job_config.parallelism.tensor_parallel_degree != 1:
            global_avg_loss, global_max_loss, global_ar_loss, global_ar_max_loss = torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)
            print("global_ar_loss",global_ar_loss)
        # else:
        #     print("self.job_config.parallelism.tensor_parallel_degree ",self.job_config.parallelism.tensor_parallel_degree )


        with Timing("benchmark_eval_step"):
            if self.step in self.lm_evaluate_steps and self.job_config.metrics.enable_lm_evaluation:
                results, flat_metrics_benchmark = self.eval_step_hellaswag()
            else:
                flat_metrics_benchmark = None
                results = None

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            val_to_log,
            global_ar_loss,
            global_ar_max_loss,
            grad_norm.item(),
            learning_rate=self.lr_schedulers.schedulers[0].get_last_lr()[0],
            flat_metrics_benchmark=flat_metrics_benchmark
        )

        

        # print("Exiting train_step() self.ft_manager._manager._quorum_future =", self.ft_manager._manager._quorum_future)

        # exit(0)


    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step,)
        logger.info(f"Training starts at step {self.step + 1}.")

        if ( self.job_config.fault_tolerance.semi_sync_method == 'diloco' and self.job_config.fault_tolerance.enable ) \
            and (self.step > 0  \
                 or ( self.job_config.checkpoint.initial_load_path is not None and self.job_config.checkpoint.enable_checkpoint)
                ):
            # if we loaded a checkpoint in semi_sync training, we don't need to recover
            logger.info("Disabling immediate recovery in semi_sync training")

            self.ft_manager._manager._init_sync = False 

            for i, frag in enumerate(self.semi_sync._fragments):
                # after loading, update parameters in the original_parameters dict
                frag.save_parameters()


        data_iterator = self.batch_generator(self.dataloader)
        # if self.step > 0:
        #     logger.info("Preemptively running the dataloader to avoid any missing samples")
        #     # preemptively run the dataloder to avoid any missing samples
        #     for i in range(10):
        #         next(data_iterator)
                
        #     logger.info("COMPLETED Preemptively running the dataloader to avoid any missing samples")

        if self.job_config.metrics.eval_only:
            results, flat_metrics_benchmark = self.eval_step_hellaswag()
            
            # Save results and flat metrics to disk using JSON with current date
            # Only save on global_rank 0 to avoid duplicate writes
            if self.job_config.global_rank == 0:
                import json
                from datetime import datetime
                
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_filename = f"eval_results_{current_date}{self.job_config.metrics.wandb_suffix}.json"
                flat_metrics_filename = f"eval_flat_metrics_{current_date}{self.job_config.metrics.wandb_suffix}.json"
                
                with open(results_filename, 'w') as f:
                    json.dump(results, f, indent=2)
                
                with open(flat_metrics_filename, 'w') as f:
                    json.dump(flat_metrics_benchmark, f, indent=2)
                
                logger.info(f"Saved evaluation results to {results_filename}")
                logger.info(f"Saved flat metrics to {flat_metrics_filename}")
            
            # Barrier to ensure all ranks wait before exiting
            torch.distributed.barrier()
            exit(0)

        with (
            maybe_enable_profiling(job_config, global_step=self.step) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config, global_step=self.step
            ) as memory_profiler,
            self.semi_sync,
        ):
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderStopIteration:
                    logger.warning("Ran out of data; last step was canceled.")
                    break


                # TODO: fix checkpointing and add this back    
                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.world_mesh,
                    )




        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Training completed")


    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}


    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]


    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()



        







if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[Trainer] = None


    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")

    exit(0)
