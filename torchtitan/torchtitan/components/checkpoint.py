# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import os
import queue
import re
import shutil
import threading
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader

from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


# For now, we will manually pop the freqs_cis buffer, as we made this permanent
# temporarily and we don't want to include it in the exported state_dict.
# Context: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model.py#L404
excluded_parameters_for_model_only = {"freqs_cis"}


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        # Exclude parameters that should not be saved
        for excluded_key in excluded_parameters_for_model_only:
            state_dict.pop(excluded_key, None)
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


@torch.no_grad()
def save_with_gc(state, checkpoint_id, process_group=None):
    dcp.save(state, checkpoint_id=checkpoint_id, process_group=process_group)
    GarbageCollection.collect("GC collection invoked by checkpointer.")


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0 or when ft_keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)

            if not 'ft-replica' in path:
                logger.info("Checkpointer is deleting %s.", path)
                
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)

            if not 'ft-replica' in path:
                logger.info(
                    "Checkpointer deleted %s in %.2f seconds.",
                    path,
                    time.monotonic() - begin,
                )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is gauranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Note: TorchFT checkpointing flow

    There are two types of checkpoints: when TorchFT is enabled: 1) the full perisistent
    checkpoint, 2) the per-replica checkpoint.

    The full perisistent checkpoint is saved by the replica with
    ``ft_manager.participating_rank() == 0``. It contains everything including the model,
    optimizer, lr_scheduler, dataloader, and train_state. Right now the full perisistent
    checkpoint is loaded by all replicas. However, we can optimize it to only load if
    there are no other alive replicas.

    The per-replica checkpoint contains only the dataloader and is saved/loaded by all
    replicas to/from the its own folder. The folder name is prefixed with the ft_replica_id.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        job_config (JobConfig): The job config used to configure the checkpointing.
        ft_manager (Optional[ft.Manager]): The FTManager from TorchFT.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        job_config: JobConfig,
        ft_manager: FTManager,
        delete_checkpoints_at_last_step: bool = False,
    ) -> None:
        ckpt_config = job_config.checkpoint
        self.enable_checkpoint = ckpt_config.enable_checkpoint
        self.checkpoint_offset = ckpt_config.checkpoint_offset
        self.ft_manager = ft_manager.manager if ft_manager.enabled else None
        self.ft_config = job_config.fault_tolerance
        
        # Validate configuration: semi_sync_method should be None when fault_tolerance is disabled
        if not ft_manager.enabled and self.ft_config.semi_sync_method is not None:
            logger.warning(
                f"Configuration mismatch: fault_tolerance.enable=False but "
                f"semi_sync_method='{self.ft_config.semi_sync_method}'. "
                f"Treating semi_sync_method as None to avoid checkpoint hangs."
            )
            # Create a modified config that treats semi_sync_method as None
            self._semi_sync_enabled = False
        else:
            self._semi_sync_enabled = self.ft_config.semi_sync_method is not None
        
        self.keep_selected_checkpoints = ckpt_config.keep_selected_checkpoints
        self.delete_checkpoints_at_last_step = delete_checkpoints_at_last_step
        print("inside CheckpointManager.__init__(), delete_checkpoints_at_last_step =", delete_checkpoints_at_last_step)

        # Always cache optimizer state dict to avoid nested collective operations during dcp.save().
        # This is critical for DDP/FSDP where get_optimizer_state_dict() is a collective operation
        # that can cause deadlocks if called inside dcp.save() (another collective).
        optimizers.init_cache_state_dict()

        if self.ft_manager:
            def state_dict():
                ret = {}
                for k, v in self.states.items():
                    if k in {
                        MODEL,
                        OPTIMIZER,
                        LR_SCHEDULER,
                        TRAIN_STATE,
                    }:
                        ret[k] = v.state_dict()
                return ret

            def load_state_dict(state_dict):
                assert state_dict is not None
                for k, v in state_dict.items():
                    self.states[k].load_state_dict(v)

            self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
        self.ft_replica_id = job_config.fault_tolerance.replica_id

        async_mode = ckpt_config.async_mode.lower()
        self.enable_staging = (
            self.enable_checkpoint and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ) or self.ft_manager

        if not self.enable_checkpoint and self.ft_manager is None:
            return

        print("dataloader", type(dataloader))
        
        # Store dataloader separately - it will be saved/loaded per-rank
        self.dataloader = dataloader
        self.global_rank = job_config.global_rank
        
        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                # NOTE: DATALOADER is NOT included here - it's saved/loaded per-rank separately
                LR_SCHEDULER: lr_schedulers,
            }
        )
        self.ft_states = {DATALOADER: dataloader} 

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None

        self.folder = os.path.join(job_config.job.dump_folder, ckpt_config.folder)

        # Checkpoint policy related fields.
        self.initial_load_path = ckpt_config.initial_load_path
        self.initial_load_model_weights_only = (
            ckpt_config.initial_load_model_weights_only
        )
        self.last_save_model_weights_only = ckpt_config.last_save_model_weights_only
        self.export_dtype = TORCH_DTYPE_MAP[ckpt_config.export_dtype]
        self.exclude_from_loading = ckpt_config.exclude_from_loading
        self.interval = ckpt_config.interval
        self.enable_first_step_checkpoint = ckpt_config.enable_first_step_checkpoint

        # Async checkpoint related fields.
        async_mode = ckpt_config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            or self.ft_manager
        ):
            self.pg = dist.new_group(backend="gloo")

        self.ft_keep_latest_k = job_config.fault_tolerance.checkpoint_keep_latest_k
        self.keep_latest_k = ckpt_config.keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(f"Unkown checkpoint async_mode {ckpt_config.async_mode}")

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable_checkpoint") and self.enable_checkpoint:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False, semi_sync: Any = None) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            None
        """

        if self.delete_checkpoints_at_last_step and last_step:


            print("inside CheckpointManager.save(), delete_checkpoints_at_last_step and last_step are True. Deleting all checkpoints.")

            if self.ft_manager:
                if os.path.exists(self.folder) and self.ft_manager.participating_rank() == 0 and dist.get_rank() == 0:
                    os.system(f"rm -r {self.folder}")
            elif not self.enable_checkpoint:
                print("inside CheckpointManager.save(), not self.enable_checkpoint and self.ft_manager is None. Not deleting any checkpoints. Returning.")
                return
            else:
                if os.path.exists(self.folder) and dist.get_rank() == 0:
                    os.system(f"rm -r {self.folder}")

            # add all checkpoints to the purge queue
            # self._purge_checkpoints_in_folder(self.folder, keep_latest_k=0, max_delete=-1)
            # count = 0
            # while not self.purge_queue.empty():
            #     time.sleep(0.1)  # Small delay to avoid busy waiting
            #     count += 1
            #     if count > 6000:
            #         logger.warning("Purge queue is not empty after 10 minutes. Continuing anyway.")
            #         break
            return

        if self.ft_manager:
            self._ft_save(curr_step)

        if not self._should_save(curr_step, last_step):
            return



        begin = time.monotonic()
        if not self.ft_manager \
            or self.ft_manager.participating_rank() == 0 \
            or self._semi_sync_enabled:
            logger.info("Saving the checkpoint (or staging if async is enabled).")
            checkpoint_id = self._create_checkpoint_id(curr_step)

            if self._semi_sync_enabled:
                checkpoint_id = os.path.join(checkpoint_id, f"ft-replica-{self.ft_replica_id}")

            self._async_wait()
            # This GC is called for async checkpoint as it is useless to do
            # GC right after async_save -- the CPU memory is not able to be
            # freed until _async_wait()

            # Save dataloader separately per-rank (all ranks save their own dataloader state)
            self._save_dataloader_per_rank(curr_step)

            if last_step:
                self._save_last_step(curr_step)
            elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    self.stager = DefaultStager(StagingOptions(True, True, True, True))
                result = dcp.async_save(
                    self.states,
                    checkpoint_id=checkpoint_id,
                    process_group=self.pg,
                    async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                    async_stager=self.stager,
                )
                self.save_future = result.upload_completion
                self.staging_future = result.staging_completion
            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                self.save_future = dcp.async_save(
                    self.states, checkpoint_id=checkpoint_id, process_group=self.pg
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")
            else:
                # Pass process_group when ft_manager is set or semi_sync is enabled
                # to ensure proper coordination with TorchFT's managed process groups.
                # self.pg may not exist when ft_manager is None and async_mode is disabled.
                pg = getattr(self, 'pg', None)
                save_with_gc(self.states, checkpoint_id=checkpoint_id, process_group=pg)
            
            # Save dataloader separately per-rank (all ranks save their own dataloader state)
            # self._save_dataloader_per_rank(curr_step)
            
            self._purge_stale_checkpoints()

            logger.info(
                "Finished saving the checkpoint (or staging if async is enabled)"
                f"in {time.monotonic() - begin:.2f} seconds."
            )
        elif self.ft_manager:
            logger.info(
                "Replica %d doesn't save checkpoint.",
                self.ft_manager.participating_rank(),
            )
            # Even if this replica doesn't save model/optimizer, it still saves its dataloader
            self._save_dataloader_per_rank(curr_step)
    


    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """

        if self.ft_manager:
            self._ft_load()

        logger.info(f"inside load(), after _ft_load()")

        if not self.enable_checkpoint:
            logger.info(f"inside load(), enable_checkpoint is False. Not loading any checkpoint. Returning False.")
            return False

        model_only = False
        if not os.path.exists(self.folder):
            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "initial_load_full_checkpoint is specified but the path is not valid."
                    )
                model_only = self.initial_load_model_weights_only
            else:
                logger.info(f"inside load(), folder: {self.folder} does not exist.")
                logger.info("inside load(), initial_load_path is not provided and the checkpoint folder does not exist. Returning False.")
                return False
        else:
            if self.initial_load_path:
                logger.info(
                    "`initial_load_path` is provided but the checkpoint folder exists. "
                    "Checkpointer will use the checkpoints from the checkpoint folder."
                )
            # if self.ft_config.semi_sync_method != None and torch.distributed.get_rank() == 0:
            #     folder = os.path.join(self.folder, f"ft-replica-{self.ft_replica_id}")
            # else:
            #     folder = self.folder
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)
            if self._semi_sync_enabled:
                # specify the replica to make sure we load diff opt states
                checkpoint_id = os.path.join(checkpoint_id, f"ft-replica-{self.ft_replica_id}")

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)

        # Load model, optimizer, lr_scheduler, train_state (NOT dataloader)
        dcp.load(states, checkpoint_id=checkpoint_id)
        logger.info(
            f"Finished loading model/optimizer/scheduler checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        
        # Load dataloader separately per-rank (each rank loads its own dataloader state)
        if not model_only and DATALOADER not in self.exclude_from_loading:
            dataloader_begin = time.monotonic()
            dataloader_loaded = self._load_dataloader_per_rank(checkpoint_id)
            if dataloader_loaded:
                logger.info(
                    f"Finished loading dataloader checkpoint in {time.monotonic() - dataloader_begin:.2f} seconds."
                )
            else:
                logger.warning(
                    "Dataloader checkpoint not loaded - this may be a checkpoint from before per-rank dataloader saving was implemented."
                )
        
        GarbageCollection.collect("GC collection for checkpoint loading.")
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            self.staging_future.result()

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)

            if not match:
                continue

            if self._semi_sync_enabled:
                metadata_flag = True
                # check that all replicas have metadata files
                for replica_id in range(self.ft_config.group_size):
                    metadata_probe = os.path.join(folder, filename, f"ft-replica-{replica_id}", ".metadata")
                    metadata_flag = metadata_flag and os.path.isfile(metadata_probe)

                if metadata_flag:
                    step_counts.append(int(match.group(1)))
            else:
                metadata_probe = os.path.join(folder, filename, ".metadata")

                if os.path.isfile(metadata_probe):
                    step_counts.append(int(match.group(1)))

        if not step_counts:
            return -1
        return max(step_counts)

    def _ft_folder(self) -> str:
        return os.path.join(self.folder, f"ft-replica-{self.ft_replica_id}")

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _create_dataloader_checkpoint_id(self, step: int, folder: str = "") -> str:
        """Create per-rank dataloader checkpoint path.
        
        Structure: step-X/dataloader/global_rank-Y/
        This allows DDP with 8 ranks and DiLoCo with 8 workers to save consistently.
        """
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}", "dataloader", f"global_rank-{self.global_rank}")

    def _save_dataloader_per_rank(self, curr_step: int) -> None:
        """Save dataloader state per-rank using no_dist=True.
        
        Dataloader state is per-rank (each rank reads different data), so we save
        each rank's state to its own path independently.
        """
        if self.dataloader is None:
            return
            
        dataloader_checkpoint_id = self._create_dataloader_checkpoint_id(curr_step)
        logger.info(f"Saving dataloader checkpoint (per-rank) to: {dataloader_checkpoint_id}")
        
        try:
            dataloader_states = {DATALOADER: self.dataloader}
            dcp.save(dataloader_states, checkpoint_id=dataloader_checkpoint_id, no_dist=True)
            logger.info(f"Finished saving dataloader checkpoint for global_rank={self.global_rank}")
        except Exception as e:
            logger.error(f"Failed to save dataloader checkpoint: {e}")
            raise

    def _load_dataloader_per_rank(self, checkpoint_id: str) -> bool:
        """Load dataloader state per-rank using no_dist=True.
        
        Args:
            checkpoint_id: The base checkpoint path (e.g., step-1000)
            
        Returns:
            bool: Whether the dataloader was loaded successfully
        """
        if self.dataloader is None:
            return False
            
        # Extract step from checkpoint_id to construct dataloader path
        step_match = re.search(r'step-(\d+)', checkpoint_id)
        if not step_match:
            logger.warning(f"Could not extract step from checkpoint_id: {checkpoint_id}")
            return False
        step = int(step_match.group(1))
        
        # Extract the base folder from checkpoint_id to support loading from initial_load_path
        step_folder_match = re.search(r'^(.+)/step-\d+', checkpoint_id)
        if step_folder_match:
            folder = step_folder_match.group(1)
        else:
            folder = self.folder
        
        dataloader_checkpoint_id = self._create_dataloader_checkpoint_id(step, folder=folder)
        
        if not os.path.isdir(dataloader_checkpoint_id):
            logger.warning(f"Dataloader checkpoint not found at: {dataloader_checkpoint_id}")
            return False
            
        logger.info(f"Loading dataloader checkpoint (per-rank) from: {dataloader_checkpoint_id}")
        
        try:
            # Initialize the dataloader iterator to ensure state_dict has the _snapshot structure
            # This is required because StatefulDataLoader has different state_dict structures
            # before and after iteration starts
            if hasattr(self.dataloader, 'dataloader'):
                # For Loader wrapper class
                _ = iter(self.dataloader.dataloader)
            else:
                _ = iter(self.dataloader)
            
            dataloader_states = {DATALOADER: self.dataloader}
            dcp.load(dataloader_states, checkpoint_id=dataloader_checkpoint_id, no_dist=True)
            logger.info(f"Finished loading dataloader checkpoint for global_rank={self.global_rank}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataloader checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _ft_save(self, step: int) -> None:
        if self.ft_config.semi_sync_method != None:
            # logger.info(f"inside _ft_save(), ft_config.semi_sync_method is not None. Not saving any checkpoint. Returning.")
            return

        begin = time.monotonic()
        self._async_wait()
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.save_future = dcp.async_save(
            self.ft_states, checkpoint_id=checkpoint_id, process_group=self.pg
        )
        self._purge_stale_ft_checkpoints(step)
        # logger.info(f"Staging ft checkpoint took {time.monotonic() - begin} secs.") # TODO: BEN left a comment here, but I don't know what it means

    def _ft_load(self) -> None:
        if self.ft_config.semi_sync_method != None:
            # logger.info(f"inside _ft_load(), ft_config.semi_sync_method is not None. Not loading any checkpoint. Returning.")
            return

        step = self._find_load_step(folder=self._ft_folder())
        if step == -1:
            return

        begin = time.monotonic()
        logger.info(f"Loading the FT checkpoint at step {step}.")
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        dcp.load(self.ft_states, checkpoint_id=checkpoint_id)
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the ft checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    def add_to_states(self, states: dict[str, Any]) -> None:
        self.states.update(states)

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        NOTE: DATALOADER is NOT included here - it's loaded separately per-rank
        via _load_dataloader_per_rank() to ensure each rank gets its own state.

        Args:
            model_only (bool): Whether to load the model only.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model weights.
        if model_only:
            sd = self.states[MODEL].state_dict()
            return sd

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states and exclude_key != DATALOADER:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        # Exclude DATALOADER from shared loading - it's loaded per-rank separately
        exclude_keys = set(self.exclude_from_loading) | {DATALOADER}
        states_to_load = {
            k: v for k, v in self.states.items() if k not in exclude_keys
        }

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving weights only at the end of the training. So
        # this won't affect preemption and training resume. We also only allow
        # dtype conversion when we are checkpoint model weights only and the
        # current dtype is not the same as the export dtype at the end of the training.

        # Determine the checkpoint path and process group for semi_sync (DiLoCo) mode
        checkpoint_id = self._create_checkpoint_id(curr_step)
        if self._semi_sync_enabled:
            checkpoint_id = os.path.join(checkpoint_id, f"ft-replica-{self.ft_replica_id}")
        pg = getattr(self, 'pg', None)

        if self.last_save_model_weights_only:
            # We update self.states to keep the model only.
            # After this update, self.states = {
            #      'tok_embeddings.weight':...,
            #      'layers.0.attention.wq.weight': ...
            # }.
            states_to_save = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states_to_save = {
                    k: v.to(self.export_dtype) for k, v in states_to_save.items()
                }
            logger.info(
                f"Saving a model weights only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
            save_with_gc(states_to_save, checkpoint_id=checkpoint_id, process_group=pg)
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            save_with_gc(self.states, checkpoint_id=checkpoint_id, process_group=pg)
            # Also save dataloader per-rank for full checkpoint
            self._save_dataloader_per_rank(curr_step)

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        if curr_step in self.keep_selected_checkpoints:
            return True

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if (curr_step) % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif self.async_mode == AsyncMode.ASYNC or self.ft_manager is not None:
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled "
                "and fault tolerance is not active."
            )

    def _purge_checkpoints_in_folder(self, folder: str, keep_latest_k: int, max_delete: int = -1) -> None:
        """Helper function to purge old checkpoints in a given folder."""
        discovered_checkpoints = []
        for filename in os.listdir(folder):
            match = re.search(r"step-(\d+)", filename)
            if match:  # Only process files that match the pattern
                path = os.path.join(folder, filename)
                discovered_checkpoints.append((int(match.group(1)), path))

        discovered_checkpoints.sort()
        if max_delete == -1:
            to_delete = discovered_checkpoints[: -1 * keep_latest_k]
        else:
            # Only delete checkpoints with step numbers smaller than max_delete
            to_delete = [(step, path) for step, path in discovered_checkpoints if step < max_delete]
            # Respect keep_latest_k limit
            to_delete = to_delete[: -1 * keep_latest_k] if len(to_delete) > keep_latest_k else []


        if self.keep_selected_checkpoints:
            # print("keeping ", [(step, path) for step, path in to_delete if step in self.keep_selected_checkpoints])
            to_delete = [(step, path) for step, path in to_delete if step not in self.keep_selected_checkpoints]

        for _, path in to_delete:
            assert self.purge_thread is not None
            self.purge_queue.put(path)

    def _purge_stale_checkpoints(self):
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
            and (not self.ft_manager or self.ft_manager.participating_rank() == 0)
        ):
            self._purge_checkpoints_in_folder(self.folder, self.keep_latest_k)
                
    def _purge_stale_ft_checkpoints(self, step: int):
        if (
            self.ft_keep_latest_k > 0
            and self.ft_manager is not None
            and os.path.isdir(self._ft_folder())
        ):
            self._purge_checkpoints_in_folder(self._ft_folder(), self.ft_keep_latest_k, max_delete=step)
