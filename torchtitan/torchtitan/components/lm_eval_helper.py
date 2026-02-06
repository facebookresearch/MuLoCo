
import math
import sys
from typing import Any, Dict, List, Optional

import contextlib
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

import lm_eval
from lm_eval.api.model import LM

from torchtitan.components.tokenizer import HuggingFaceTokenizer, build_hf_tokenizer
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.utils import device_module, device_type

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Holds parsed evaluation metrics for a single task."""
    task_name: str
    alias: Optional[str] = None
    acc: Optional[float] = None
    acc_stderr: Optional[float] = None
    acc_norm: Optional[float] = None
    acc_norm_stderr: Optional[float] = None


def parse_lm_eval_results(
    results: Dict[str, Any],
    tasks: Optional[List[str]] = None,
    include_subtasks: bool = True,
) -> Dict[str, TaskResult]:
    """
    Parse LM Evaluation Harness results dict to extract accuracy metrics.

    Args:
        results: The full results dict returned by lm_eval.simple_evaluate().
        tasks: Optional list of specific task names to extract. If None, extracts all.
        include_subtasks: If True, include subtasks (e.g., mmlu_formal_logic under mmlu).

    Returns:
        Dict mapping task names to TaskResult objects containing:
          - acc (accuracy)
          - acc_stderr (accuracy standard error)
          - acc_norm (normalized accuracy, if available)
          - acc_norm_stderr (normalized accuracy standard error, if available)

    Example:
        >>> parsed = parse_lm_eval_results(results)
        >>> print(parsed['arc_challenge'].acc)
        0.23122866894197952
        >>> print(parsed['arc_challenge'].acc_norm)
        0.2687713310580205
    """
    parsed_results: Dict[str, TaskResult] = {}

    # The actual task results are nested under 'results' key
    task_results = results.get("results", {})

    for task_name, metrics in task_results.items():
        # Filter by task list if provided
        if tasks is not None:
            # Check if this task matches any requested task
            matches = False
            for requested_task in tasks:
                if task_name == requested_task:
                    matches = True
                    break
                # Check if it's a subtask (e.g., mmlu_formal_logic for mmlu)
                if include_subtasks and task_name.startswith(f"{requested_task}_"):
                    matches = True
                    break
            if not matches:
                continue

        # Skip subtasks if not requested
        if not include_subtasks:
            # Check if this is a subtask by looking at the alias
            alias = metrics.get("alias", "")
            if alias.startswith("  -"):  # Subtasks have aliases like "  - formal_logic"
                continue

        # Extract metrics - handle both regular floats and np.float64
        def safe_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        task_result = TaskResult(
            task_name=task_name,
            alias=metrics.get("alias"),
            acc=safe_float(metrics.get("acc,none")),
            acc_stderr=safe_float(metrics.get("acc_stderr,none")),
            acc_norm=safe_float(metrics.get("acc_norm,none")),
            acc_norm_stderr=safe_float(metrics.get("acc_norm_stderr,none")),
        )

        parsed_results[task_name] = task_result

    return parsed_results


def get_flat_metrics_dict(
    results: Dict[str, Any],
    tasks: Optional[List[str]] = None,
    include_subtasks: bool = False,
    prefix: str = "eval/",
) -> Dict[str, float]:
    """
    Convert LM eval results to a flat dict suitable for logging to wandb/tensorboard.

    Args:
        results: The full results dict returned by lm_eval.simple_evaluate().
        tasks: Optional list of specific task names to extract.
        include_subtasks: If True, include subtasks in the output.
        prefix: Prefix to add to all metric keys (e.g., "eval/" for wandb).

    Returns:
        Flat dict with keys like "eval/arc_challenge/acc", "eval/arc_challenge/acc_norm"

    Example:
        >>> metrics = get_flat_metrics_dict(results)
        >>> wandb.log(metrics, step=step)
    """
    parsed = parse_lm_eval_results(results, tasks=tasks, include_subtasks=include_subtasks)
    flat_metrics: Dict[str, float] = {}

    for task_name, task_result in parsed.items():
        base_key = f"{prefix}{task_name}"

        if task_result.acc is not None:
            flat_metrics[f"{base_key}/acc"] = task_result.acc
        if task_result.acc_stderr is not None:
            flat_metrics[f"{base_key}/acc_stderr"] = task_result.acc_stderr
        if task_result.acc_norm is not None:
            flat_metrics[f"{base_key}/acc_norm"] = task_result.acc_norm
        if task_result.acc_norm_stderr is not None:
            flat_metrics[f"{base_key}/acc_norm_stderr"] = task_result.acc_norm_stderr

    return flat_metrics



class _DistributedStub:
    """
    Minimal accelerator-like object implementing the subset of methods
    that `lm_eval.evaluator.evaluate` expects when running in distributed
    mode:

      - `gather(tensor)`
      - `wait_for_everyone()`

    It uses `torch.distributed` under the hood when a process group is
    initialized; otherwise it falls back to single-process behavior.
    """

    def __init__(self, device: torch.device, debug: bool = False) -> None:
        self.device = device
        self.debug = debug

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather a scalar tensor from all ranks into a 1D tensor of length world_size.
        """
        if self.debug:
            print("in _DistributedStub() Entering gather available:", dist.is_available(), "initialized:", dist.is_initialized(), "world_size:", dist.get_world_size())
        if not dist.is_available() or not dist.is_initialized():
            return tensor.new_tensor([tensor.item()])

        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor.new_tensor([tensor.item()])

        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        if self.debug:
            print("in _DistributedStub() Exiting gather")
        return torch.stack(gather_list)

    def wait_for_everyone(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def all_reduce_max(self, value: int) -> int:
        """
        Perform an all-reduce MAX operation on an integer value across all ranks.
        
        Args:
            value: The local integer value to reduce.
            
        Returns:
            The maximum value across all ranks.
        """
        if not dist.is_available() or not dist.is_initialized():
            return value

        world_size = dist.get_world_size()
        if world_size == 1:
            return value

        value_tensor = torch.tensor(value, dtype=torch.long, device=self.device)
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
        return int(value_tensor.item())


class TorchTitanDistributedStub:
    """
    Accelerator-like object for TorchTitan that uses mesh-based distributed
    operations compatible with various parallelism configurations (DP, FSDP, CP, etc.).

    This stub properly handles:
      - Data parallel reductions across dp_cp mesh dimension
      - Optional fault tolerance (DiLoCo) reductions across ft_pg

    It implements the subset of methods that `lm_eval.evaluator.evaluate` expects:
      - `gather(tensor)`
      - `wait_for_everyone()`
    """

    def __init__(
        self,
        device: torch.device,
        dp_cp_mesh: "DeviceMesh",
        ft_pg: Optional[dist.ProcessGroup] = None,
        debug: bool = False,
    ) -> None:
        """
        Args:
            device: The device tensors are on.
            dp_cp_mesh: The flattened dp_cp mesh from TorchTitan's world_mesh["dp_cp"].
            ft_pg: Optional fault tolerance process group for DiLoCo reductions.
                   When enabled, reductions happen across both dp_cp_mesh and ft_pg.
            debug: If True, print debug statements.
        """
        self.device = device
        self.dp_cp_mesh = dp_cp_mesh
        self.ft_pg = ft_pg
        self.debug = debug

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather a scalar tensor from all ranks in the dp_cp mesh (and ft_pg if enabled)
        into a 1D tensor.
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor.new_tensor([tensor.item()])

        # First gather within the dp_cp mesh
        dp_cp_group = self.dp_cp_mesh.get_group()
        dp_cp_size = self.dp_cp_mesh.size()
        
        # DEBUG: Print group info
        global_rank = dist.get_rank()
        print(f"[DEBUG gather] global_rank={global_rank}, "
              f"dp_cp_group={dp_cp_group}, "
              f"dp_cp_size={dp_cp_size}, "
              f"dp_cp_group_rank={dist.get_rank(dp_cp_group)}", 
              flush=True)

        if dp_cp_size == 1 and self.ft_pg is None:
            return tensor.new_tensor([tensor.item()])

        # Gather within dp_cp mesh
        gather_list = [torch.zeros_like(tensor) for _ in range(dp_cp_size)]
        dist.all_gather(gather_list, tensor, group=dp_cp_group)
        result = torch.stack(gather_list)

        # If ft_pg is enabled, also gather across replicas
        if self.ft_pg is not None:
            ft_size = self.ft_pg.size()
            if ft_size > 1:
                print(f"[DEBUG gather] global_rank={global_rank}, "
                      f"about to ft_pg all_gather, ft_size={ft_size}", 
                      flush=True)
                # All-gather the already gathered results across ft_pg
                ft_gather_list = [torch.zeros_like(result) for _ in range(ft_size)]
                dist.all_gather(ft_gather_list, result, group=self.ft_pg)
                result = torch.cat(ft_gather_list, dim=0)

        return result

    def wait_for_everyone(self) -> None:
        """Barrier across dp_cp mesh and ft_pg if enabled."""
        if dist.is_available() and dist.is_initialized():
            # DEBUG: Print group info to understand the mismatch
            dp_cp_group = self.dp_cp_mesh.get_group()
            global_rank = dist.get_rank()
            dp_cp_local_rank = self.dp_cp_mesh.get_local_rank()
            print(f"[DEBUG wait_for_everyone] global_rank={global_rank}, "
                  f"dp_cp_local_rank={dp_cp_local_rank}, "
                  f"dp_cp_group={dp_cp_group}, "
                  f"dp_cp_group_size={dist.get_world_size(dp_cp_group)}, "
                  f"dp_cp_group_rank={dist.get_rank(dp_cp_group)}", 
                  flush=True)
            
            dist.barrier(group=dp_cp_group)
            
            if self.ft_pg is not None:
                print(f"[DEBUG wait_for_everyone] global_rank={global_rank}, "
                      f"ft_pg={self.ft_pg}, "
                      f"ft_pg_size={self.ft_pg.size()}", 
                      flush=True)
                # Must specify device_ids to use GPU/NCCL backend - ft_pg is a TorchFT 
                # ManagedProcessGroup that doesn't have a CPU backend
                dist.barrier(group=self.ft_pg, device_ids=[device_module.current_device()])

    def all_reduce_max(self, value: int) -> int:
        """
        Perform an all-reduce MAX operation on an integer value across
        the dp_cp mesh (and ft_pg if enabled).

        Args:
            value: The local integer value to reduce.

        Returns:
            The maximum value across all ranks.
        """
        if not dist.is_available() or not dist.is_initialized():
            return value

        value_tensor = torch.tensor(value, dtype=torch.long, device=self.device)

        # Reduce across dp_cp mesh first
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX, group=self.dp_cp_mesh.get_group())

        # Then reduce across ft_pg if enabled
        if self.ft_pg is not None:
            dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX, group=self.ft_pg)

        return int(value_tensor.item())

    def _gpu_all_gather_object(self, obj: Any, group: dist.ProcessGroup) -> List[Any]:
        """
        All-gather objects using GPU tensors instead of CPU tensors.
        
        This works around the limitation that TorchFT's ManagedProcessGroup 
        doesn't support CPU tensor operations (which dist.gather_object uses internally).
        
        Args:
            obj: The object to gather from this rank.
            group: The process group to use for communication.
            
        Returns:
            List of objects gathered from all ranks.
        """
        import pickle
        
        world_size = dist.get_world_size(group)
        
        # Serialize object to bytes
        obj_bytes = pickle.dumps(obj)
        obj_size = len(obj_bytes)
        
        # Create GPU tensor for size communication
        size_tensor = torch.tensor([obj_size], dtype=torch.long, device=self.device)
        size_list = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
        
        # All-gather sizes using GPU tensors
        dist.all_gather(size_list, size_tensor, group=group)
        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes)
        
        # Pad local bytes to max_size and convert to GPU tensor
        padded_bytes = obj_bytes + b'\x00' * (max_size - obj_size)
        # Convert bytes to uint8 tensor on GPU
        local_tensor = torch.frombuffer(bytearray(padded_bytes), dtype=torch.uint8).to(self.device)
        
        # All-gather the byte tensors
        gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=self.device) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, local_tensor, group=group)
        
        # Deserialize objects
        result = []
        for i, t in enumerate(gathered_tensors):
            obj_bytes = bytes(t[:sizes[i]].cpu().numpy())
            result.append(pickle.loads(obj_bytes))
        
        return result

    def gather_object(
        self,
        obj: Any,
        object_gather_list: Optional[List[Any]],
        dst: int = 0,
    ) -> None:
        """
        Gather arbitrary picklable objects from all ranks in the dp_cp mesh 
        (and ft_pg if enabled) to the destination rank.

        This is compatible with torch.distributed.gather_object but works with
        TorchFT process groups that are not visible to the default torch.distributed.

        Args:
            obj: The object to gather from this rank.
            object_gather_list: On the destination rank, a list to populate with 
                               gathered objects (one per rank). Must be None on 
                               non-destination ranks.
            dst: The destination rank (in global terms, typically 0).
        """
        if not dist.is_available() or not dist.is_initialized():
            if object_gather_list is not None:
                object_gather_list[0] = obj
            return

        dp_cp_group = self.dp_cp_mesh.get_group()
        dp_cp_size = self.dp_cp_mesh.size()
        dp_cp_rank = self.dp_cp_mesh.get_local_rank()
        
        # DEBUG: Print group info
        global_rank = dist.get_rank()
        print(f"[DEBUG gather_object] global_rank={global_rank}, "
              f"dp_cp_group={dp_cp_group}, "
              f"dp_cp_size={dp_cp_size}, "
              f"dp_cp_rank={dp_cp_rank}, "
              f"dp_cp_group_rank={dist.get_rank(dp_cp_group)}", 
              flush=True)

        # Determine if this rank is the destination within the dp_cp mesh
        # dst=0 in global terms means dp_cp_rank=0 AND ft_rank=0
        is_dp_cp_dst = (dp_cp_rank == 0)

        # First, gather within dp_cp mesh to dp_cp_rank 0
        if is_dp_cp_dst:
            dp_cp_gather_list = [None] * dp_cp_size
        else:
            dp_cp_gather_list = None

        print(f"[DEBUG gather_object] global_rank={global_rank}, about to call dist.gather_object", flush=True)
        dist.gather_object(
            obj=obj,
            object_gather_list=dp_cp_gather_list,
            dst=0,
            group=dp_cp_group,
        )
        print(f"[DEBUG gather_object] global_rank={global_rank}, dist.gather_object done", flush=True)

        # If ft_pg is enabled, also gather across ft replicas
        if self.ft_pg is not None and is_dp_cp_dst:
            ft_size = dist.get_world_size(self.ft_pg)
            ft_rank = dist.get_rank(self.ft_pg)

            if ft_size > 1:
                # Use GPU-based all_gather for ft_pg to avoid CPU backend issues
                # TorchFT's ManagedProcessGroup doesn't support CPU tensor operations
                ft_gathered = self._gpu_all_gather_object(dp_cp_gather_list, self.ft_pg)

                # Flatten the nested lists on the global destination rank
                if ft_rank == 0 and object_gather_list is not None:
                    idx = 0
                    for replica_samples in ft_gathered:
                        for sample in replica_samples:
                            object_gather_list[idx] = sample
                            idx += 1
            else:
                # ft_size == 1, just copy dp_cp results
                if ft_rank == 0 and object_gather_list is not None:
                    for i, sample in enumerate(dp_cp_gather_list):
                        object_gather_list[i] = sample
        elif self.ft_pg is None and is_dp_cp_dst:
            # No ft_pg, just copy dp_cp results to object_gather_list
            if object_gather_list is not None:
                for i, sample in enumerate(dp_cp_gather_list):
                    object_gather_list[i] = sample


class TorchTitanLM(LM):
    """
    LM Evaluation Harness wrapper for a TorchTitan decoder-only Transformer model.

    This class adapts a `torchtitan.models.llama3.model.Transformer` and a
    `HuggingFaceTokenizer` into the `lm_eval.api.model.LM` interface described in
    `docs/model_guide.md`:

      - `loglikelihood`
      - `loglikelihood_rolling`
      - `generate_until`
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        global_rank: int,
        global_world_size: int,
        max_seq_len: Optional[int] = None,
        max_batch_size: int = 1,
        # TorchTitan-specific parameters for parallelism support
        world_mesh: Optional["DeviceMesh"] = None,
        parallel_dims: Optional[Any] = None,
        train_context: Optional[Any] = None,
        maybe_enable_amp: Optional[Any] = None,
        job_config: Optional[Any] = None,
        ft_pg: Optional[dist.ProcessGroup] = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.global_rank = global_rank
        self.debug = debug

        # Use the tokenizer passed in from the training pipeline (e.g. TikTokenizer
        # or HuggingFaceTokenizer) so that token IDs are consistent with the model.
        self.tokenizer = tokenizer if not isinstance(tokenizer, str) else build_llama3_tokenizer(tokenizer)
        self.device = next(model.parameters()).device

        # Sequence length configuration
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        else:
            # Fallback to model config if available
            self.max_seq_len = getattr(
                getattr(model, "model_args", None), "max_seq_len", 2048
            )

        # Batch size for evaluation
        self.max_batch_size = max_batch_size

        # TorchTitan parallelism support
        self.world_mesh = world_mesh
        self.parallel_dims = parallel_dims
        self.train_context = train_context
        self.maybe_enable_amp = maybe_enable_amp if maybe_enable_amp is not None else contextlib.nullcontext()
        self.job_config = job_config
        self.ft_pg = ft_pg

        self._world_size = global_world_size
        self._rank = global_rank

        # Provide a stub "accelerator" to satisfy evaluator's expectations.
        # Use TorchTitan stub if mesh is provided, otherwise fall back to basic stub
        # The dp_cp mesh is created via _flatten() in ParallelDims.build_mesh, so we need
        # to check if it's accessible as a child mesh, not in mesh_dim_names directly.
        if world_mesh is not None:
            try:
                dp_cp_mesh = world_mesh["dp_cp"]
                self.accelerator = TorchTitanDistributedStub(
                    device=self.device,
                    dp_cp_mesh=dp_cp_mesh,
                    ft_pg=ft_pg,
                    debug=self.debug,
                )
            except (KeyError, RuntimeError):
                # Fall back to basic stub if dp_cp mesh is not available
                self.accelerator = _DistributedStub(self.device, debug=self.debug)
        else:
            self.accelerator = _DistributedStub(self.device, debug=self.debug)

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    def _encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Encode text into token IDs using the provided tokenizer.

        Supports both:
          - TorchTitan TikTokenizer: encode(text, bos=..., eos=...)
          - HF-style tokenizers: encode(text), with manual BOS/EOS handling.
        """
        tok = self.tokenizer

        # Try TikTokenizer-style signature first (bos/eos keyword args).
        try:
            return tok.encode(text, bos=add_bos, eos=add_eos)
        except TypeError:
            # Fall back to a HF-style tokenizer: encode(text),
            # then prepend/append BOS/EOS if available.
            ids = tok.encode(text)

            # Add BOS if requested and tokenizer exposes a BOS token id.
            bos_id = getattr(tok, "bos_token_id", None)
            if add_bos and bos_id is not None:
                ids = [bos_id] + ids

            # Add EOS if requested and tokenizer exposes an EOS token id.
            eos_id = getattr(tok, "eos_token_id", None)
            if add_eos and eos_id is not None:
                ids = ids + [eos_id]

            return ids

    def _get_optional_cp_context(self, inputs: torch.Tensor):
        """
        Create context parallel context if CP is enabled.

        Args:
            inputs: Input tensor to include in CP buffers.

        Returns:
            Context manager for context parallelism, or None if CP is disabled.
        """
        if (
            self.parallel_dims is not None
            and self.parallel_dims.cp_enabled
            and self.world_mesh is not None
            and self.job_config is not None
        ):
            return dist_utils.create_context_parallel_ctx(
                cp_mesh=self.world_mesh["cp"],
                cp_buffers=[inputs, self.model.freqs_cis],
                cp_seq_dims=[1, 0],
                cp_no_restore_buffers={inputs},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
        return None

    def _forward_logits(self, token_ids: List[int]) -> torch.Tensor:
        """
        Run the underlying model and return logits.

        Args:
            token_ids: List of token IDs (length L)

        Returns:
            logits: Tensor of shape [1, L, vocab_size]
        """
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        input_ids_shape = input_ids.shape

        # Get context parallel context if enabled
        optional_cp_ctx = self._get_optional_cp_context(input_ids)

        with torch.no_grad():
            if self.debug:
                print("in _forward_logits() input_ids shape:", input_ids_shape)
            if self.train_context is not None and optional_cp_ctx is not None:
                with self.train_context(optional_cp_ctx):
                    with self.maybe_enable_amp:
                        logits = self.model(input_ids)
            elif self.train_context is not None:
                with self.train_context(None):
                    with self.maybe_enable_amp:
                        logits = self.model(input_ids)
            else:
                with self.maybe_enable_amp:
                    logits = self.model(input_ids)
            if self.debug:
                print("after model forward in _forward_logits() ")
        return logits

    def _create_progress_bar(self, total: int, method_name: str):
        """Create a progress bar for tracking evaluation progress using tqdm."""
        try:
            # raise ImportError("tqdm is not available")
            from tqdm import tqdm
            return tqdm(total=total, desc=method_name, unit="item")
        except ImportError:
            # Fallback to simple progress bar if tqdm is not available
            class SimpleProgressBar:
                def __init__(self, total, prefix):
                    self.total = total
                    self.prefix = prefix
                    self.current = 0
                    
                def update(self, n=1):
                    self.current += n
                    if self.total > 0:
                        percent = (self.current / self.total) * 100
                        bar_length = 40
                        filled_length = int(bar_length * self.current // self.total)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)
                        print(f'\r{self.prefix}: |{bar}| {self.current}/{self.total} ({percent:.1f}%)', end='', flush=True)
                        if self.current >= self.total:
                            print()  # New line when complete
                            
            return SimpleProgressBar(total, f"{method_name}")

    # -------------------------------------------------------------------------
    # Required LM interface
    # -------------------------------------------------------------------------
    def loglikelihood(self, requests):
        """
        Each request has Instance.args == (context: str, continuation: str).

        For a given (context, continuation) pair, we:
          1. Tokenize context and context+continuation.
          2. Align continuation tokens with the model's next-token predictions.
          3. Sum log-probabilities of the continuation tokens.
          4. Report whether the continuation is the greedy sequence.

        Returns:
            list[tuple[float, bool]]
        """
        if self.debug:
            print("Entering loglikelihood")
        # Pre-allocate results list with None to maintain original order
        results: List[tuple[float, bool]] = [None] * len(requests)
        if self.global_rank == 0:
            progress_bar = self._create_progress_bar(len(requests), "loglikelihood")


        if self.debug:
            print("before instance for loop - preprocessing all requests")

        # Preprocess all requests to extract tokens and metadata
        preprocessed = []
        for idx, instance in enumerate(requests):
            context, continuation = instance.args

            # Encode with an explicit BOS token for stability.
            ctx_tokens = self._encode(context, add_bos=True, add_eos=False)
            full_tokens = self._encode(context + continuation, add_bos=True, add_eos=False)

            # Derive continuation tokens by slicing off the context prefix.
            if len(full_tokens) < len(ctx_tokens):
                # Degenerate edge case; fall back to treating all of full_tokens as continuation.
                cont_tokens = full_tokens
                ctx_len = 0
            else:
                ctx_len = len(ctx_tokens)
                cont_tokens = full_tokens[ctx_len:]

            # Handle empty continuation case immediately
            if len(cont_tokens) == 0:
                results[idx] = (0.0, True)
                if self.global_rank == 0:
                    progress_bar.update()
                continue

            # Truncate to max_seq_len from the left if needed.
            if len(full_tokens) > self.max_seq_len:
                excess = len(full_tokens) - self.max_seq_len
                full_tokens = full_tokens[-self.max_seq_len:]
                # Adjust context length accordingly.
                ctx_len = max(0, ctx_len - excess)

            preprocessed.append({
                'idx': idx,
                'full_tokens': full_tokens,
                'ctx_len': ctx_len,
                'cont_tokens': cont_tokens,
                'original_len': len(full_tokens),
            })

        if self.debug:
            print(f"Preprocessed {len(preprocessed)} requests for batching with max_batch_size={self.max_batch_size}")

        # Find the maximum sequence length across ALL requests to avoid recompilation
        # for torch compile / flex attention (same padded length for all batches)
        if len(preprocessed) > 0:
            found_max_seq_len = max(item['original_len'] for item in preprocessed)
        else:
            found_max_seq_len = 0

        # Synchronize found_max_seq_len across all ranks to ensure consistent padded_len
        # This prevents NCCL errors from mismatched tensor shapes in distributed evaluation
        found_max_seq_len = self.accelerator.all_reduce_max(found_max_seq_len)

        # Check if found max exceeds user-specified max_seq_len
        if found_max_seq_len > self.max_seq_len:
            raise ValueError(
                f"Found maximum sequence length ({found_max_seq_len}) exceeds "
                f"the user-specified max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or use shorter sequences."
            )

        # Use the found max, capped by max_seq_len
        padded_len = min(self.max_seq_len, found_max_seq_len)
        if self.debug:
            print(f"Global padded_len for all batches: {padded_len} (found_max={found_max_seq_len}, max_seq_len={self.max_seq_len})")

        # Process in batches
        for batch_start in range(0, len(preprocessed), self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, len(preprocessed))
            batch = preprocessed[batch_start:batch_end]
            batch_size = len(batch)

            if self.debug:
                print(f"Processing batch {batch_start // self.max_batch_size + 1}, size={batch_size}")

            # Pad all sequences in the batch to the global padded_len
            batch_token_ids = []
            for item in batch:
                tokens = item['full_tokens'].copy() if isinstance(item['full_tokens'], list) else list(item['full_tokens'])
                # Pad with zeros to padded_len
                if len(tokens) < padded_len:
                    tokens = tokens + [0] * (padded_len - len(tokens))
                batch_token_ids.append(tokens)

            # Forward pass for the entire batch
            if self.debug:
                print(f"Before forward logits, batch shape: [{batch_size}, {padded_len}]")
            input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)

            # Get context parallel context if enabled
            optional_cp_ctx = self._get_optional_cp_context(input_ids)

            with torch.no_grad():
                if self.train_context is not None and optional_cp_ctx is not None:
                    with self.train_context(optional_cp_ctx):
                        with self.maybe_enable_amp:
                            logits = self.model(input_ids)  # [B, L, V]
                elif self.train_context is not None:
                    with self.train_context(None):
                        with self.maybe_enable_amp:
                            logits = self.model(input_ids)  # [B, L, V]
                else:
                    with self.maybe_enable_amp:
                        logits = self.model(input_ids)  # [B, L, V]
                log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
            if self.debug:
                print("After forward logits")

            # Process each sample in the batch to extract results
            for i, item in enumerate(batch):
                ctx_len = item['ctx_len']
                cont_tokens = item['cont_tokens']
                original_idx = item['idx']

                # With standard decoder-only training, logits[t] predict token_ids[t+1].
                # Continuation tokens correspond to positions:
                #   start_pos = ctx_len - 1
                #   end_pos   = ctx_len + len(cont_tokens) - 1   (exclusive)
                start_pos = max(ctx_len - 1, 0)
                end_pos = start_pos + len(cont_tokens)

                token_positions = log_probs[i, start_pos:end_pos, :]  # [K, V]

                target_ids = torch.tensor(cont_tokens, dtype=torch.long, device=self.device)
                token_logprobs = token_positions.gather(
                    dim=-1, index=target_ids.unsqueeze(-1)
                ).squeeze(-1)  # [K]

                total_ll = float(token_logprobs.sum().item())

                # Check greediness: whether each continuation token is argmax at its position.
                greedy_ids = token_positions.argmax(dim=-1)  # [K]
                is_greedy = bool(torch.equal(greedy_ids, target_ids))

                results[original_idx] = (total_ll, is_greedy)

                if self.global_rank == 0:
                    progress_bar.update()
                    sys.stdout.flush()

        if self.debug:
            print("Exiting loglikelihood")

        return results

    def loglikelihood_rolling(self, requests):
        """
        Each request has Instance.args == (text: str,).

        We compute the log-likelihood of the entire sequence conditioned on BOS,
        using the standard next-token prediction convention.

        Returns:
            list[float]
        """
        results: List[float] = []
        progress_bar = self._create_progress_bar(len(requests), "loglikelihood_rolling")

        for instance in requests:
            (text,) = instance.args
            tokens = self._encode(text, add_bos=True, add_eos=False)

            if len(tokens) <= 1:
                results.append(0.0)
                progress_bar.update()
                continue

            # Inputs are all tokens except the last; targets are the next tokens.
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            logits = self._forward_logits(input_tokens)  # [1, L-1, V]
            log_probs = F.log_softmax(logits, dim=-1)

            target_ids = torch.tensor(target_tokens, dtype=torch.long, device=self.device)
            token_logprobs = log_probs[0, :, :].gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)

            total_ll = float(token_logprobs.sum().item())
            results.append(total_ll)
            progress_bar.update()

        return results

    def generate_until(self, requests):
        """
        Each request has Instance.args == (context: str, gen_kwargs: dict).

        We implement simple greedy generation up to `max_gen_toks` tokens, stopping
        early if any of the `until` strings (if provided) appears in the decoded text.

        Returns:
            list[str]: one generated continuation per request.
        """
        outputs: List[str] = []
        progress_bar = self._create_progress_bar(len(requests), "generate_until")

        for instance in requests:
            context, gen_kwargs = instance.args
            gen_kwargs = gen_kwargs or {}

            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 32))
            until = gen_kwargs.get("until") or []

            # Work on a copy of token ids; ensure BOS for stability.
            tokens = self._encode(context, add_bos=True, add_eos=False)

            generated_tokens: List[int] = []

            for _ in range(max_gen_toks):
                # Truncate to model's max sequence length from the left if needed.
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[-self.max_seq_len :]

                logits = self._forward_logits(tokens)
                next_token_logits = logits[0, -1, :]
                next_token_id = int(next_token_logits.argmax(dim=-1).item())

                tokens.append(next_token_id)
                generated_tokens.append(next_token_id)

                # Decode current generation and check stopping criteria.
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                if any(stop_str in text for stop_str in until):
                    break

            continuation_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=False
            )
            outputs.append(continuation_text)
            progress_bar.update()

        return outputs


def build_llama3_tokenizer(tokenizer_root: str) -> HuggingFaceTokenizer:
    """
    Convenience helper to build a tokenizer for Meta-Llama-3 style models.

    Args:
        tokenizer_root: Path to a directory containing tokenizer files. For example:
            `/home/.../torchtitan/assets/tokenizer/Meta-Llama-3.1-8B`
    """
    return build_hf_tokenizer(tokenizer_root)


def evaluate_torchtitan_hellaswag(
    model: nn.Module,
    tokenizer: Any,
    max_seq_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    limit: Optional[int] = 100,
    num_fewshot: int = 0,
    log_samples: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    global_rank: int = 0,
    global_world_size: int = 1,
    # TorchTitan-specific parameters for parallelism support
    world_mesh: Optional["DeviceMesh"] = None,
    parallel_dims: Optional[Any] = None,
    train_context: Optional[Any] = None,
    maybe_enable_amp: Optional[Any] = None,
    job_config: Optional[Any] = None,
    ft_pg: Optional[dist.ProcessGroup] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the LM Evaluation Harness on the `hellaswag` task using a TorchTitan model.

    This function wraps the provided model and tokenizer in a `TorchTitanLM`
    and calls `lm_eval.simple_evaluate` with a fixed task list `["hellaswag"]`.

    Args:
        model: A TorchTitan decoder-only Transformer instance (e.g. llama3.Transformer).
        tokenizer: A HuggingFaceTokenizer built for this model.
        max_seq_len: Optional override for the maximum sequence length. If None,
            we will try to read `model.model_args.max_seq_len` or default to 2048.
        limit: Optional limit on the number of Hellaswag examples to evaluate
            (useful for quick tests).
        num_fewshot: Number of few-shot examples to use (default 0).
        log_samples: Whether to log per-sample outputs in the results dict.
        random_seed, numpy_random_seed, torch_random_seed: Seeds forwarded to
            `simple_evaluate` for reproducibility.
        global_rank: The global rank of the process.
        world_mesh: TorchTitan's world mesh for parallelism support.
        parallel_dims: ParallelDims object for checking parallelism configuration.
        train_context: Context manager for loss parallel / compiled autograd / CP.
        maybe_enable_amp: Context manager for automatic mixed precision.
        job_config: Job configuration for CP rotate method.
        ft_pg: Fault tolerance process group for DiLoCo reductions (optional).

    Returns:
        A results dictionary in the standard LM Eval Harness format.
    """
    print(f"global_rank: {global_rank}")
    print(f"global_world_size: {global_world_size}")
    # print(f"world_mesh: {world_mesh}")
    # print(f"parallel_dims: {parallel_dims}")
    # print(f"train_context: {train_context}")
    # print(f"maybe_enable_amp: {maybe_enable_amp}")
    # print(f"job_config: {job_config}")
    # print(f"ft_pg: {ft_pg}")
    # print(f"debug: {debug}")
    lm = TorchTitanLM(
        model=model, 
        tokenizer=tokenizer, 
        max_seq_len=max_seq_len, 
        max_batch_size=max_batch_size,
        global_rank=global_rank,
        global_world_size=global_world_size,
        world_mesh=world_mesh,
        parallel_dims=parallel_dims,
        train_context=train_context,
        maybe_enable_amp=maybe_enable_amp,
        job_config=job_config,
        ft_pg=ft_pg,
        debug=debug,
    )


    with torch.no_grad():

        #'mmlu','hellaswag','piqa','arc_challenge','arc_easy','winogrande','openbookqa','pubmedqa','qnli','qqp','rte','squad','superglue','wsc','wic'],

        results = lm_eval.simple_evaluate(
            model=lm,
            cache_requests=True,
            tasks=[
                'mmlu',
                'hellaswag',
                'piqa',
                'arc_challenge',
                'arc_easy',
                'winogrande',
                'openbookqa',
                'arc_challenge',
                'arc_easy',
                # 'winogrande',
                # 'pubmedqa',
                # 'qnli',
                # 'qqp',
                # 'rte',
                # 'squad',
                # 'superglue',
                # 'wsc',
                # 'wic',
            ],
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=log_samples,
            random_seed=random_seed,
            numpy_random_seed=numpy_random_seed,
            torch_random_seed=torch_random_seed,
        )


    if global_rank == 0:

        flat_metrics = get_flat_metrics_dict(
                            results,
                            include_subtasks=True,
                            prefix="eval_tasks/",
                        )
    else:
        flat_metrics = None




    return results, flat_metrics


