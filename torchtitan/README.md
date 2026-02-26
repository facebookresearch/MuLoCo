# MuLoCo: Muon is a practical inner optimizer for DiLoCo

This repository contains the training code for the MuLoCo paper, built on top of [TorchTitan](https://github.com/pytorch/torchtitan) — a PyTorch native platform for training generative AI models.

[![arXiv](https://img.shields.io/badge/arXiv-2505.23725-b31b1b.svg)](https://arxiv.org/abs/2505.23725)

## Abstract

DiLoCo is a powerful framework for training large language models (LLMs) under networking constraints with advantages for increasing parallelism and accelerator utilization in data center settings. Despite significantly reducing communication frequency, however, DiLoCo's communication steps still involve all-reducing a complete copy of the model's parameters. In this work, we investigate the effectiveness of standard compression methods including Top-k sparsification and quantization for reducing the communication overhead of DiLoCo when paired with two local optimizers (AdamW and Muon). Our experiments pre-training decoder-only transformer language models reveal that leveraging Muon as the inner optimizer for DiLoCo along with an error-feedback accumulator allows to aggressively compress the communicated delta to 2-bits with next to no performance degradation. **Crucially, MuLoCo (Muon inner optimizer DiLoCo) significantly outperforms DiLoCo while communicating 8X less and having identical memory complexity.**

## Acknowledgments

This codebase is built upon **TorchTitan**, developed by the PyTorch team. We extend their framework with implementations for DiLoCo, MuLoCo, and various communication-efficient training methods. For details on TorchTitan's core features and parallelism techniques, please see the [original TorchTitan repository](https://github.com/pytorch/torchtitan).

We also acknowledge the following contributions that informed our implementation:
- **[Dion](https://github.com/microsoft/dion)** by Microsoft Research — for the initial distributed implementation of DiLoCo and related methods
- **[TorchTitan Muon PR #1521](https://github.com/pytorch/torchtitan/pull/1521)** — for the original Data Parallel Muon implementation in TorchTitan

## Repository Structure

### Configuration Files

All training configurations are located in:

```
config/
├── base.py                          # Base configuration
├── ddp/                             # DDP (Distributed Data Parallel) configs
│   ├── 1gpu_test.py
│   ├── 8gpu_test_3B.py
│   ├── 8gpu_test_15B.py
│   ├── adamw.py                     # DDP with AdamW optimizer
│   └── muon.py                      # DDP with Muon optimizer
└── diloco/                          # DiLoCo/MuLoCo configs
    ├── 2x_4gpu_diloco.py            # DiLoCo baseline
    ├── 2x_4gpu_diloco_quantized_outer.py
    ├── 2x_4gpu_diloco_streaming.py
    ├── 2x_4gpu_gpa.py               # Gradient Projection Average
    ├── 2x_4gpu_muloco.py            # MuLoCo (Muon inner optimizer)
    ├── 2x_4gpu_muloco_quantized_outer.py
    └── 2x_4gpu_muloco_streaming.py
```

### Job Launch Scripts

Job launch scripts are located in:

```
jobs/
├── ddp/                             # DDP training jobs
│   ├── run_1gpu_test.sh
│   └── run_8gpu_test_adamw.sh
├── diloco/                          # Interactive DiLoCo/MuLoCo jobs
│   └── debug_8workers_x1gpu.sh
└── muloco_paper/                    # Paper experiment launchers
    ├── ddp_baselines_h100.py
    ├── diloco_muloco_no-compression_h100.py
    ├── inner_sweep_distributed_quantized_streaming.py
    ├── inner_sweep_distributed_quantized_topk.py
    ├── model_map.py
    └── generated_job_files/
```

## Quick Start

### Interactive Training (Single Node)

For **DDP training** within an interactive job (8 GPUs):

```bash
# Request an interactive allocation first, then run:
./jobs/ddp/run_8gpu_test_adamw.sh
```

For **DiLoCo/MuLoCo training** with 8 workers (1 GPU each) within an interactive job:

```bash
# Request an interactive allocation first, then run:
./jobs/diloco/debug_8workers_x1gpu.sh
```

This launches 8 independent workers plus the `torchft_lighthouse` coordinator, simulating distributed local SGD training on a single node.

## Reproducing Paper Experiments (SLURM Batch Jobs)

For large-scale experiments replicating our paper results, use the job generation scripts in:

```
jobs/muloco_paper/
```

### Paper Experiment Launchers

| Script | Description |
|--------|-------------|
| `ddp_baselines_h100.py` | Generates and launches DDP baseline experiments (AdamW and Muon optimizers) for comparison against DiLoCo/MuLoCo |
| `diloco_muloco_no-compression_h100.py` | Generates and launches DiLoCo and MuLoCo experiments without gradient compression |
| `inner_sweep_distributed_quantized_streaming.py` | Generates experiments with streaming quantization (error-feedback quantized outer optimizer) |
| `inner_sweep_distributed_quantized_topk.py` | Generates experiments with TopK sparsification for gradient compression |
| `model_map.py` | Model configurations including step counts, batch sizes, and optimal hyperparameters for various Llama model sizes |

### Running Paper Experiments

1. **Modify cluster-specific parameters** in the launcher scripts before running:

```python
# Example from diloco_muloco_no-compression_h100.py
partition_args = f"""--partition h100 --account optim --qos optim_high"""
```

You **must** update these values to match your SLURM cluster configuration:

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--partition` | SLURM partition/queue name | `h100`, `gpu`, `compute` |
| `--account` | Billing/allocation account | `optim`, `myproject`, `default` |
| `--qos` | Quality of Service tier | `normal`, `high`, `optim_high` |

2. **Generate and launch jobs**:

```bash
cd jobs/muloco_paper/
python diloco_muloco_no-compression_h100.py
```

This will:
- Generate SLURM job scripts in `generated_job_files/`
- Automatically submit jobs with `sbatch` while respecting node limits
- Monitor running jobs to avoid exceeding allocation quotas

### Model Configurations

The `model_map.py` file contains pre-configured settings for various Llama model sizes used in our experiments:
| Model Flavor | Parameters | Layers | Heads | QKV Dim | Hidden Dim | Token Budget | HP Sweep |
|--------------|------------|--------|-------|---------|------------|--------------|----------|
| `llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf` | 150M | 6 | 4 | 512 | 1,408 | 3B | ✓ |
| `llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf` | 416M | 12 | 8 | 1,024 | 2,816 | 8.16B | ✓ |
| `llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf` | 914M | 18 | 12 | 1,536 | 4,224 | 18.12B | ✓ |
| `llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf` | 1.76B | 24 | 16 | 2,048 | 5,632 | 35.23B | ✓ |
| `llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf` | 3.07B | 30 | 20 | 2,560 | 7,040 | 61.4B | ✓ |
| `llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf` | 15.23B | 54 | 36 | 4,608 | 12,672 | 304.6B | ✗ |



## Key Configuration Options

### DiLoCo/MuLoCo-specific settings (in config files):

```python
fault_tolerance = {
    'enable': True,                    # Enable distributed local SGD
    'semi_sync_method': 'diloco',      # Algorithm: 'diloco', 'gpa', etc.
    'sync_steps': 30,                  # Inner steps between synchronization
    'group_size': 8,                   # Number of workers
    'replica_id': 0,                   # Worker ID (0 to group_size-1)
}

outer_optimizer = {
    'kwargs': {
        'momentum': 0.9,
        'lr': 0.8,
        'nesterov': True,
    },
    'class_name': 'sgd'
}
```

### Optimizer selection:

- **DiLoCo**: Uses AdamW as the inner optimizer
- **MuLoCo**: Uses Muon as the inner optimizer (set `optimizer.name = 'Muon'`)

## Running Jobs on a SLURM Cluster

For a detailed guide on configuring `generate_job_file.py` for your cluster (QOS settings, NCCL tuning, log paths, etc.) and understanding the full job generation pipeline, see [RUNNING_JOBS.md](./RUNNING_JOBS.md).

## Installation and Environment Setup

For installation instructions (UV, Rust, protoc, virtual environment setup), see the [main README](../README.md).

Once installed, activate the environment:

```bash
export MULOCO_PATH=/path/to/MuLoCo
source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan
```

## Citation

If you use this code, please cite both our work and TorchTitan:

```bibtex
@article{therien2025muloco,
  title={MuLoCo: Muon is a practical inner optimizer for DiLoCo},
  author={Th{\'e}rien, Benjamin and Huang, Xiaolong and Rish, Irina and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2505.23725},
  year={2025},
  url={https://arxiv.org/abs/2505.23725}
}

@inproceedings{liang2025torchtitan,
  title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
  author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

## License

This code is released under a BSD 3 license, following the original TorchTitan license. See [LICENSE](./LICENSE) for details.
