# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Any, TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, device_module, device_type

import os.path as osp
from torchtitan.distributed.utils import dist_broadcast_string, dist_mean

if TYPE_CHECKING:
    from torchtitan.protocols.train_spec import BaseModelArgs


# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class Timing:

    # Static dictionaries to store run times and historical stats
    run_times_dict = defaultdict(list)  # Stores the elapsed times for each named timer
    historical_stats = defaultdict(lambda: {"mean": [], "std": []})  # Stores the historical mean and std for each named timer

    def __init__(self,name,list=[]):
        self.name = name
        self.list = list

    def __enter__(self):
        # print("entering timing", self.name)
        self.start = time.time()
        return self  # This allows us to use "as x" in the with statement

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        duration = self.end - self.start
        Timing.run_times_dict[self.name].append(duration)
        self.list.append(duration)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: str | None = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()


class DummyLogger(BaseLogger):
    """A dummy logger that does nothing, used for non-logging ranks."""

    def __init__(self):
        self.wandb = None # set to non to prevent calling close()

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass



def get_ckpt_dirs(ckpt_dir, checkpoint_name):
    a = os.listdir(ckpt_dir)
    keep = []
    for x in a:
        # 8 for wandb id +1 for underscore
        if osp.isdir(osp.join(ckpt_dir, x)) and x[9:] == checkpoint_name:
            keep.append(x)
    return keep


def get_resume_ckpt(ckpt_dir, checkpoint_name):

    if not osp.exists(ckpt_dir):
        logger.info("[Info] No existing checkpoint found. Starting from scratch.")
        return None

    dirs = get_ckpt_dirs(ckpt_dir, checkpoint_name)

    if len(dirs) == 0:
        logger.info("No existing checkpoint found. Starting from scratch.")
        return None
    elif len(dirs) == 1:
        logger.info("Found 1 checkpoint. Loading from {}".format(dirs[0]))
        return osp.join(ckpt_dir, dirs[0])
    else:
        raise NotImplementedError("Found {} checkpoints. Loading from {}".format(len(dirs), dirs[0]))



    # ckpt_path, suffix = get_ckpt_to_load(ckpt_dir, dirs)
    # print("[Info] Loading checkpoint from {}".format(ckpt_path))
    # return ckpt_path


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(self, 
        job_config: JobConfig, 
        tag: str | None = None,
        ckpt_path: str = None
    ):
        # Import wandb here to avoid startup import
        import wandb
        self.job_config = job_config
        self.wandb = wandb
        self.tag = tag
        self.last_logged_step = -1  # Track the last step we successfully logged


        ckpt_path = get_resume_ckpt(job_config.job.dump_folder, job_config.checkpoint_name)

        if job_config.global_rank == 0:
            if ckpt_path is not None:
                self.log_dir = self.resume_wandb(job_config, tag, ckpt_path)
            else:
                self.log_dir = self.init_wandb(job_config, tag)
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            while ckpt_path is None:
                ckpt_path = get_resume_ckpt(job_config.job.dump_folder, job_config.checkpoint_name)
                time.sleep(1)

            self.log_dir = ckpt_path
            self.wandb = DummyLogger()
            logger.info("using a dummy logger on non-rank 0")



    def init_wandb(self, job_config: JobConfig, tag: str | None = None):
        run = self.wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            # dir=os.environ.get("WANDB_DIR", log_dir), WANDB_DIR is used by default
            group="fixed_"+job_config.wandb_name,
            config=job_config.to_dict(),
        )
        logger.info("WandB logging enabled")

        return os.path.join(job_config.job.dump_folder, f"{run.id}_" + job_config.checkpoint_name)
        


    def resume_wandb(self, job_config: JobConfig, tag: str | None = None, ckpt_path: str = None):
        print("init using resume",ckpt_path.split('/')[-1][:8])
        print("ckpt_path", ckpt_path)
        self.wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            # dir=os.environ.get("WANDB_DIR", log_dir), WANDB_DIR is used by default
            group="fixed_"+job_config.wandb_name,
            config=job_config.to_dict(),
            resume='must',
            id=ckpt_path.split('/')[-1][:8],
        )
        logger.info("WandB logging enabled")

        return ckpt_path


    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb is None:
            return

        if self.job_config.global_rank == 0:
            if self.wandb.run is not None:
                self.wandb.finish()

                


def ensure_pp_loss_visible(
    parallel_dims: ParallelDims, job_config: JobConfig, color: Color
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if job_config.parallelism.pipeline_parallel_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if job_config.parallelism.pipeline_parallel_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)

def _build_metric_logger(
    job_config: JobConfig, 
    parallel_dims: ParallelDims, 
    tag: str | None = None,
    ft_pg: Any | None = None,
    mesh: Any | None = None,
    device: torch.device | None = None,
) -> BaseLogger:
    """
    Build an appropriate metric logger based on configuration.
    """
    metrics_config = job_config.metrics

    # Log initial config state
    logger.debug(
        f"Building logger with config: wandb={metrics_config.enable_wandb}, "
        f"tensorboard={metrics_config.enable_tensorboard}"
    )

    # Check if any logging backend is enabled
    has_logging_enabled = (
        metrics_config.enable_tensorboard or metrics_config.enable_wandb
    )

    # Determine if this rank should log
    should_log = has_logging_enabled
    if (not metrics_config.save_for_all_ranks) and should_log:
        metrics_rank = _get_metrics_rank(parallel_dims, job_config)
        should_log = torch.distributed.get_rank() == metrics_rank

    logger.debug(
        f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
    )



    # Create loggers in priority order
    if metrics_config.enable_wandb:
        logger.debug("Attempting to create WandB logger")
        try:
            wandb_logger = WandBLogger(job_config, tag)
        except Exception as e:
            if "No module named 'wandb'" in str(e):
                logger.error(
                    "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                )

                raise e
            else:
                logger.error(f"Failed to create WandB logger: {e}")
                raise e

        # print("before dist_mean")
        # ten = torch.tensor([5.0]).to(device)
        # print(f"Rank {job_config.global_rank}: {ten}")
        # print(f"Rank {job_config.global_rank}: mesh={mesh}, ft_pg={ft_pg}")

        # output = dist_mean(ten, mesh, ft_pg)



        # print("dist_mean", output)
        # print("\n\noutput check\n\n")

        # if job_config.global_rank == 0:
        #     dump_folder = dist_broadcast_string(
        #         wandb_logger.log_dir, 0, job_config.global_rank, mesh, ft_pg, device)
        #     os.makedirs(dump_folder, exist_ok=True)
        # else:
        #     dump_folder = dist_broadcast_string(
        #         wandb_logger.log_dir, 0, job_config.global_rank, mesh, ft_pg, device)

        job_config.job.dump_folder = wandb_logger.log_dir
        logger.info(f"wandb_logger.log_dir: {wandb_logger.log_dir}")
        dump_folder = wandb_logger.log_dir

        base_log_dir = os.path.join(
            dump_folder, 
            metrics_config.save_tb_folder, 
            datetime.now().strftime("%Y%m%d-%H%M")
        )

        if metrics_config.save_for_all_ranks:
            base_log_dir = os.path.join(
                base_log_dir, f"rank_{torch.distributed.get_rank()}"
            )


        if not should_log:
            logger.debug("Returning BaseLogger due to should_log=False")
            return BaseLogger()
        else:
            return wandb_logger
        
        
        
    else:
        raise NotImplementedError("Only Wandb logging is supported at this time")

    # if metrics_config.enable_tensorboard:
    #     logger.debug("Creating TensorBoard logger")
    #     return TensorBoardLogger(base_log_dir, tag)

    # logger.debug("No loggers enabled, returning BaseLogger")
    # return BaseLogger()


class MetricsProcessor:
    """Metrics processor to processes the metrics and log metrics.

    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        tag (Optional[str]): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    logger: BaseLogger
    parallel_dims: ParallelDims
    job_config: JobConfig
    device_memory_monitor: DeviceMemoryMonitor
    color: utils.NoColor | utils.Color

    gpu_peak_flops: int
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: str | None = None,
        mesh: Any | None = None,
        ft_pg: Any | None = None,
        device: torch.device | None = None,
    ):
        self.logger = _build_metric_logger(job_config, parallel_dims, tag, mesh=mesh, ft_pg=ft_pg, device=device)
        self.parallel_dims = parallel_dims
        self.job_config = job_config
        self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = (
            utils.NoColor()
            if job_config.metrics.disable_color_printing
            else utils.Color()
        )

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.lr_schedulers = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.job_config.metrics.log_freq == 0

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        val_metrics: dict[str, Any] | None,
        global_ar_loss: float | None, # loss across DP ranks
        global_ar_max_loss: float | None, # loss across DP ranks
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
        learning_rate: float | None = None,
        flat_metrics_benchmark: dict[str, Any] | None = None,
    ):
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12

        time_end_to_end = time_delta / self.job_config.metrics.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "loss_metrics/global_ar_avg_loss": global_ar_loss,
            "loss_metrics/global_ar_max_loss": global_ar_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "tflops": tflops,
            "mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
            "timing/optimizers.step": Timing.run_times_dict["optimizers.step"][-1],
            "timing/dist_mean_train": Timing.run_times_dict["dist_mean_train"][-1],
            "learning_rate": learning_rate,
            "iteration": step,
            # "timing/eval_step": Timing.run_times_dict["eval_step"],
        }

        if flat_metrics_benchmark is not None:
            metrics.update(flat_metrics_benchmark)

        if val_metrics is not None:
            metrics.update(val_metrics)

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        def log():
            logger.info(
                f"{color.red}step: {step:2}  "
                f"{color.green}loss: {global_avg_loss:7.4f}  "
                f"{color.orange}grad_norm: {grad_norm:7.4f}  "
                f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                f"{color.white}data T: {time_data_loading:7.4f}s  "
                f"{color.blue}tps: {round(tps):,}  "
                f"{color.cyan}tflops: {tflops:,.2f}  "
                f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                f"{color.red}opt: {Timing.run_times_dict['optimizers.step'][-1]:.4f}s  "
                f"{color.orange}loss AR: {Timing.run_times_dict['dist_mean_train'][-1]:.4f}s  "
                # f"{color.yellow}eval: {Timing.run_times_dict['eval_step']:.4f}s  "
            )

        if self.job_config.metrics.log_all_ranks:
            log()
        elif self.job_config.global_rank == 0:
            log()

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def close(self):
        self.logger.close()


def build_metrics_processor(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    model_args: "BaseModelArgs | None" = None,
    tag: str | None = None,
    mesh: Any | None = None,
    ft_pg: Any | None = None,
    device: torch.device | None = None,
) -> MetricsProcessor:
    """Create a metrics processor.

    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        model_args (BaseModelArgs | None): Model-specific arguments. Defaults to None.
        tag (str | None): Tag to use for TensorBoard or WandB. Defaults to None.

    Returns:
        MetricsProcessor: A metrics processor.
    """
    return MetricsProcessor(job_config, parallel_dims, tag, mesh=mesh, ft_pg=ft_pg, device=device)
