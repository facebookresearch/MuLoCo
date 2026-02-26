# Running Jobs on a SLURM Cluster

This document explains how to use the job generation system to launch MuLoCo training experiments on a SLURM cluster.

## Overview

The job generation pipeline has three layers:

```
Job Scripts (jobs/muloco_paper/*.py)
    │
    ▼
generate_job_file.py  ──►  Generated .sh files (SLURM batch scripts)
    │                              │
    ▼                              ▼
Launch Scripts                 sbatch <script>.sh
(launch_distributed.py /
 launch_diloco_multinode.py)
```

1. **`generate_job_file.py`** — Assembles SLURM batch scripts from cluster settings, setup prefixes, and launch commands.
2. **Job scripts** (`jobs/muloco_paper/*.py`) — Define experiment sweeps (learning rates, batch sizes, model sizes, etc.) and call `generate_job_file.py` to produce `.sh` files, then optionally submit them via `sbatch`.
3. **Launch scripts** (`launch_distributed.py`, `launch_diloco_multinode.py`) — Called from within the generated `.sh` scripts at runtime to orchestrate the TorchFT lighthouse and training replicas.

## Configuring `generate_job_file.py` for Your Cluster

Before running any jobs, you must configure `generate_job_file.py` with your cluster's SLURM settings. The file contains:

### 1. QOS/Partition Definitions

At the top of the file, define your cluster's QOS (Quality of Service) configurations as dictionaries. Each entry maps a QOS name to its SLURM account, partition, and memory:

```python
# Example: define your cluster's QOS settings
MY_CLUSTER_GPU = {
    "account": "my-account",       # SLURM account name (--account / -A)
    "partition": "gpu",            # SLURM partition (--partition)
    "qos": "normal",              # SLURM QOS (--qos)
    "mem": "500G",                 # Memory per node (--mem)
}

MY_CLUSTER_HIGH_PRIORITY = {
    "account": "my-account",
    "partition": "gpu",
    "qos": "high",
    "mem": "500G",
}

# Register them in the QOS map
QOS_DEFAULT_MAP = {
    "normal": MY_CLUSTER_GPU,
    "high": MY_CLUSTER_HIGH_PRIORITY,
}
```

### 2. CPUs and Memory per Node

Configure the CPU and memory resources available on your nodes for each QOS:

```python
QOS_CPUS_PER_NODE = {
    "normal": 64,    # Total CPUs available per node
    "high": 128,
}

MEM_PER_NODE = {
    "normal": "500G",
    "high": "1000G",
}
```

### 3. Setup Prefix

The setup prefix is a shell script fragment that runs at the beginning of each job. It sets up the environment, discovers node IPs, and configures NCCL. You need to customize this for your cluster:

```python
SETUP_PREFIX = """source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan
export WANDB_PROJECT="my_project"

LOG_PREFIX="[DILOCO]"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "$LOG_PREFIX Nodes: ${nodes[@]}"
echo "$LOG_PREFIX Node IP: $head_node_ip"
export LOGLEVEL=INFO
export RUST_BACKTRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add any cluster-specific NCCL settings here:
# export NCCL_BUFFSIZE=8388608
# export NCCL_CROSS_NIC=1
# export NCCL_NVLS_ENABLE=1
"""
```

For **AWS-based clusters** with EFA networking, you will additionally need:
```python
# EFA settings for AWS
export FI_PROVIDER="efa"
export NCCL_IB_DISABLE=1
export FI_EFA_USE_DEVICE_RDMA=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
```

### 4. SLURM Header Defaults

The `make_slurm_header()` function has defaults for email, output paths, etc. Update these for your environment:

```python
def make_slurm_header(
    job_name="diloco",
    partition="gpu",
    account="my-account",
    qos="normal",
    nodes=2,
    gpus_per_node=8,
    ntasks_per_node=9,
    cpus_per_task=14,
    mem="500G",
    time="36:00:00",
    mail_type="BEGIN,END,FAIL",
    mail_user="you@example.com",                    # Your email
    output="/path/to/logs/%j.out",                  # Your log directory
    error="/path/to/logs/%j.err"):                  # Your log directory
```

### 5. Log Directory in Launch Scripts

The launch scripts (`launch_distributed.py`, `launch_diloco_multinode.py`) write per-replica logs. Update the `log_dir` path inside these files:

```python
log_dir = f"/path/to/your/logs/multinode_log/{SLURM_JOB_ID}"
```

## Three Types of Training Jobs

### DDP (Data-Parallel) Baselines

Standard distributed data-parallel training using `torchrun` + `srun`. Each node runs one process group.

**Generated launch command:**
```bash
srun --cpus-per-task=<N> torchrun \
  --nnodes <NODES> --nproc_per_node <GPUS> \
  --rdzv_id 101 --rdzv_backend c10d \
  --rdzv_endpoint "${head_node_ip}:29500" \
  ./torchtitan/train.py --job.config_file <CONFIG> --cfg_options <EXTRA_ARGS>
```

**Config files:** `config/ddp/adamw.py`, `config/ddp/muon.py`

**Usage flag:** `--use-ddp`

### DiLoCo / MuLoCo (Single-Node Replicas)

Each SLURM node runs multiple fault-tolerant replicas via TorchFT. `launch_distributed.py` starts a lighthouse process, then launches each replica with its own `torchrun` and a subset of GPUs.

**Architecture:**
```
Node 0:  [lighthouse] [replica 0 (GPU 0)] [replica 1 (GPU 1)] ... [replica 7 (GPU 7)]
Node 1:  [replica 8 (GPU 0)] [replica 9 (GPU 1)] ... [replica 15 (GPU 7)]
...
```

**Config files:** `config/diloco/2x_4gpu_diloco.py`, `config/diloco/2x_4gpu_muloco.py`, etc.

**Usage:** no `--use-ddp` flag, `--nodes-per-replica 1` (default)

### DiLoCo / MuLoCo (Multi-Node Replicas)

Each replica spans multiple nodes. `launch_diloco_multinode.py` groups nodes into replica chunks and launches one `torchrun` per chunk with cross-node rendezvous.

**Architecture (4 nodes, 2 nodes per replica = 2 replicas):**
```
Replica 0: [Node 0 (8 GPUs)] + [Node 1 (8 GPUs)]  ←  16 GPUs
Replica 1: [Node 2 (8 GPUs)] + [Node 3 (8 GPUs)]  ←  16 GPUs
Lighthouse: runs on Node 0
```

**Usage flag:** `--nodes-per-replica 2` (or any value > 1)

## Writing a Job Script

Job scripts live in `jobs/muloco_paper/` and programmatically call `generate_job_file.py`. Here is a minimal example:

```python
import os

OUTPUT_DIR = "jobs/my_experiment/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_flavor = "llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf"

# Sweep over learning rates
for i, lr in enumerate([0.03125, 0.015625, 0.007812]):
    extra_args = f"""\
lr_scheduler.warmup_steps=500 \
model.flavor={model_flavor} \
optimizer.lr={lr} \
training.global_batch_size=512 \
training.local_batch_size=16 \
training.steps=8160"""

    cmd = f"""python generate_job_file.py \
--config-file config/ddp/muon.py \
--job-name muon_lr{lr} \
--qos normal \
--nodes 4 \
--gpus-per-node 8 \
--time 24:00:00 \
--use-ddp \
--extra-args "{extra_args}" \
--output-file {OUTPUT_DIR}/job_{i}.sh"""

    os.system(cmd)

# Submit all generated jobs
for f in sorted(os.listdir(OUTPUT_DIR)):
    os.system(f"sbatch {os.path.join(OUTPUT_DIR, f)}")
```

For DiLoCo/MuLoCo jobs, remove `--use-ddp` and add replica configuration:

```python
cmd = f"""python generate_job_file.py \
--config-file config/diloco/2x_4gpu_muloco.py \
--job-name muloco_k16 \
--qos normal \
--nodes 16 \
--gpus-per-node 8 \
--num-ft-replicas-per-node 8 \
--nodes-per-replica 1 \
--time 48:00:00 \
--extra-args "{extra_args}" \
--output-file {OUTPUT_DIR}/muloco_job.sh"""
```

## `generate_job_file.py` Reference

```
usage: generate_job_file.py [-h]
  --config-file CONFIG_FILE     Training config file path (required)
  --qos QOS                     SLURM QOS (must be a key in QOS_DEFAULT_MAP)
  --nodes NODES                 Number of SLURM nodes
  --gpus-per-node GPUS          GPUs per node (default: 8)
  --time TIME                   Wall time limit (default: 36:00:00)
  --use-ddp                     Use DDP mode (torchrun + srun) instead of TorchFT
  --num-ft-replicas-per-node N  FT replicas per node (DiLoCo mode, default: 8)
  --nodes-per-replica N         Nodes per replica (multi-node mode, default: 1)
  --extra-args ARGS             Training config overrides passed via --cfg_options
  --output-file FILE            Write generated script to file (otherwise prints to stdout)
  --job-name NAME               SLURM job name
  --mail-user EMAIL             Email for SLURM notifications
  --output PATH                 SLURM stdout log path (use %j for job ID)
  --error PATH                  SLURM stderr log path (use %j for job ID)
```

## Model Map

`jobs/muloco_paper/model_map.py` contains lookup tables used by the job scripts:

| Map | Purpose |
|-----|---------|
| `MODEL_STEP_MAP` | Training steps for each (model_flavor, global_batch_size) pair (Chinchilla-optimal) |
| `LBS_MAP` | Maximum local batch size per GPU for each model size |
| `USE_DP_PARALLELISM_MAP` | Whether a model fits in a single GPU (True) or needs FSDP sharding (False) |
| `OPTIMAL_WD_MAP` | Tuned weight decay per model size and optimizer type |

## Quick Start Checklist

1. Edit `generate_job_file.py`:
   - Define your QOS configs in `QOS_DEFAULT_MAP`
   - Set `QOS_CPUS_PER_NODE` and `MEM_PER_NODE`
   - Update the setup prefix with your cluster's environment and NCCL settings
   - Update default email and log paths in `make_slurm_header()`
2. Edit `launch_distributed.py` and `launch_diloco_multinode.py`:
   - Update the `log_dir` path
3. Create a job script or use one from `jobs/muloco_paper/` as a template
4. Run `cd torchtitan && python jobs/muloco_paper/your_script.py`
5. Generated `.sh` files appear in the output directory and are submitted automatically
