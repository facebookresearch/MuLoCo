# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.



if __name__ == "__main__":

    import sys
    from pathlib import Path

    # Add the project root to the path to import torchtitan modules
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    

from dataclasses import dataclass
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split="train", streaming=True)

def _load_dclm_dataset(dataset_path: str):
    """Load DCLM dataset with default configuration."""
    return load_dataset(dataset_path, split="train", streaming=True)

def _load_dclm_test_dataset(dataset_path: str):
    """Load DCLM test dataset: first 10B tokens."""
    return load_dataset(dataset_path, split="train", streaming=True)

def _load_dclm_train_dataset(dataset_path: str):
    """Load DCLM train dataset: after first 10B tokens."""
    return load_dataset(dataset_path, split="train", streaming=True)

def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]

def _process_dclm_text(sample: dict[str, Any]) -> str:
    """Process DCLM dataset sample text."""
    return sample["text"]

@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable

# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=_load_c4_dataset,
        text_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        text_processor=_process_c4_text,
    ),
    "dclm": DatasetConfig(
        path="mlfoundations/dclm-baseline-1.0-parquet",
        loader=_load_dclm_train_dataset,
        text_processor=_process_dclm_text,
    ),
    "dclm_test": DatasetConfig(
        path="mlfoundations/dclm-baseline-1.0-parquet",
        loader=_load_dclm_test_dataset,
        text_processor=_process_dclm_text,
    ),
}

def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor

class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        # Handle both map-style and iterable datasets
        if isinstance(ds, Dataset):
            # Map-style dataset - use built-in sharding
            if dp_world_size > 1:
                self._data = ds.shard(num_shards=dp_world_size, index=dp_rank)
            else:
                self._data = ds
        else:
            # Iterable/streaming dataset - use HF distributed splitting
            self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    from torchtitan.components.tokenizer import Tokenizer
    from torchtitan.config_manager import JobConfig
    
    def test_dataset_format(dataset_name: str, max_samples: int = 3):
        """Test that a dataset produces the expected output format."""
        print(f"\n=== Testing {dataset_name.upper()} dataset ===")
        
        # Create a simple tokenizer for testing
        class SimpleTokenizer:
            def encode(self, text: str, bos: bool = True, eos: bool = True) -> list[int]:
                # Simple character-based tokenization for testing
                tokens = [1] if bos else []  # BOS token
                tokens.extend([ord(c) % 1000 for c in text[:100]])  # Simple char encoding
                if eos:
                    tokens.append(2)  # EOS token
                return tokens
        
        tokenizer = SimpleTokenizer()
        
        # Create dataset
        dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_path=None,
            tokenizer=tokenizer,
            seq_len=64,  # Small seq_len for testing
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
        )
        
        print(f"Dataset created successfully: {dataset.dataset_name}")
        
        # Test a few samples
        sample_count = 0
        for sample in dataset:
            if sample_count >= max_samples:
                break
            
            # Handle both tuple and dict return types
            if isinstance(sample, tuple):
                sample_data, labels = sample
            else:
                # If it's a dict, extract the relevant fields
                sample_data = sample.get('input_ids', sample.get('data', sample))
                labels = sample.get('labels', sample_data)
            
            print(f"Sample {sample_count + 1}:")
            print(f"  Sample type: {type(sample)}")
            print(f"  Sample keys (if dict): {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
            
            # Handle different data types
            if hasattr(sample_data, 'shape'):
                print(f"  Input shape: {sample_data.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Input tokens (first 10): {sample_data[:10].tolist()}")
                print(f"  Labels (first 10): {labels[:10].tolist()}")
            else:
                print(f"  Input data: {sample_data}")
                print(f"  Labels: {labels}")
            print()
            sample_count += 1
    
    # Test both C4, DCLM train, and DCLM test datasets
    test_dataset_format("c4", max_samples=2)
    test_dataset_format("dclm", max_samples=2)
    test_dataset_format("dclm_test", max_samples=2)
    
    print("Dataset testing completed!")
