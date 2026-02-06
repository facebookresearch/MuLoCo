import os
import random
from typing import List, Optional

import lm_eval
import torch
import torch.distributed as dist
from lm_eval.api.model import LM
from lm_eval.utils import setup_logging


class DistributedStub:
    """
    Minimal stand-in for an `accelerate.Accelerator` object, providing just the
    methods that `lm_eval.evaluator.evaluate` expects for multi-process runs:
      - `gather(tensor)`
      - `wait_for_everyone()`

    It uses `torch.distributed` under the hood.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather a scalar tensor from all ranks into a 1D tensor of length world_size,
        matching the semantics used in the evaluator.
        """
        if not dist.is_available() or not dist.is_initialized():
            # Single-process fallback: wrap the value in a length-1 tensor.
            return tensor.new_tensor([tensor.item()])

        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor.new_tensor([tensor.item()])

        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        return torch.stack(gather_list)

    def wait_for_everyone(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


class RandomLM(LM):
    """
    A fake language model that implements the LM interface from lm_eval.api.model.LM.

    - `loglikelihood` and `loglikelihood_rolling` return random log-likelihoods.
    - `generate_until` returns random text sampled from a simple character vocabulary.

    This is intended only for wiring up and testing the evaluation pipeline; it does
    not use any real model weights.
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        max_gen_toks: int = 32,
    ) -> None:
        super().__init__()
        # Simple character-level vocabulary; you can change this if you like.
        self.vocab = vocab or list("abcdefghijklmnopqrstuvwxyz ")
        self.max_gen_toks = max_gen_toks

    def _random_logprob(self, length: int) -> float:
        """
        Produce a random log-probability that roughly scales with sequence length.

        This has no semantic meaning; it's just to satisfy the interface.
        """
        per_token = random.uniform(-10.0, 0.0)
        return per_token * max(1, length)

    def loglikelihood(self, requests):
        """
        Each request has Instance.args == (context: str, continuation: str).
        We return a list of (logprob, is_greedy) tuples.
        """
        results: list[tuple[float, bool]] = []
        for instance in requests:
            context, continuation = instance.args
            _ = context  # unused; kept to document the interface
            ll = self._random_logprob(len(continuation))
            is_greedy = False
            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        """
        Each request has Instance.args == (text: str,).
        We return a list of log-likelihoods, one per request.
        """
        results: list[float] = []
        for instance in requests:
            (text,) = instance.args
            ll = self._random_logprob(len(text))
            results.append(ll)
        return results

    def generate_until(self, requests):
        """
        Each request has Instance.args == (context: str, gen_kwargs: dict).
        We ignore `context` and just generate random text from `self.vocab`,
        respecting `max_gen_toks` and simple `until` stopping strings.
        """
        outputs: list[str] = []
        for instance in requests:
            context, gen_kwargs = instance.args
            _ = context  # unused; kept to document the interface
            gen_kwargs = gen_kwargs or {}

            max_gen_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            until = gen_kwargs.get("until")

            tokens = [random.choice(self.vocab) for _ in range(max_gen_toks)]
            text = "".join(tokens)

            # Naively respect stopping strings by truncating at the first match.
            if until:
                for stop in until:
                    idx = text.find(stop)
                    if idx != -1:
                        text = text[:idx]
                        break

            outputs.append(text)

        return outputs


def init_distributed() -> tuple[int, int, torch.device]:
    """
    Initialize torch.distributed if launched with torchrun.

    Returns:
        rank, world_size, device
    """
    if not dist.is_available():
        return 0, 1, torch.device("cpu")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def main():
    """
    Instantiate the RandomLM and run a small Hellaswag evaluation to verify
    that the model interface is correctly wired into lm_eval.

    When launched via `torchrun --nproc-per-node=N test.py`, this will run
    evaluation in parallel across N processes using torch.distributed.
    """
    setup_logging("INFO")

    rank, world_size, device = init_distributed()

    lm = RandomLM()
    # Wire up basic distributed attributes expected by the evaluator.
    lm._rank = rank
    lm._world_size = world_size
    lm.device = device
    lm.accelerator = DistributedStub(device)

    # Use a small `limit` so this runs quickly; you can increase/remove it later.
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=10,
        log_samples=False,
    )

    if rank == 0:
        print("Evaluation results on hellaswag with RandomLM (distributed):")
        print(results)


if __name__ == "__main__":
    main()