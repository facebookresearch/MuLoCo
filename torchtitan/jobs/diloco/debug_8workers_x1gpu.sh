#!/bin/bash
#SBATCH --job-name=2gpu_dil_debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=8
#SBATCH --mem=1900G
#SBATCH --partition=learn
#SBATCH -A parq
#SBATCH --qos=parq_high
#SBATCH --time=80:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=btherien@meta.com
#SBATCH --output=/home/btherien/logs/diloco/%j.out
#SBATCH --error=/home/btherien/logs/diloco/%j.err


source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan

# export WANDB_API_KEY=local-38f7f7a0bedbbafcbc6092cca0341b3f8be5dba0
# wandb login --host https://fairwandb.org
# export WANDB_PROJECT="diloco_debug"


export WANDB_PROJECT="dil_nemotroncc_v2"

export NCCL_BUFFSIZE=8388608 
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_CROSS_NIC=1
export NCCL_NVLS_ENABLE=1
export TORCHDYNAMO_VERBOSE=1
export CUDA_LAUNCH_BLOCKING=1

ARG_CONFIG_PATH="config/diloco/2x_4gpu_diloco.py"
ARG_CONFIG_PATH="config/diloco/2x_4gpu_muloco.py"


# llama3-w1024-d12-h8
RUST_BACKTRACE=1 \
torchft_lighthouse \
--min_replicas 8 \
--quorum_tick_ms 100 \
--join_timeout_ms 10000 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=0 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=0 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=1 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=1 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=2 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=2 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=3 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=3 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=4 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=4 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=5 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=5 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=6 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=6 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1 &

NGPU=1 \
CUDA_VISIBLE_DEVICES=7 \
CONFIG_FILE=$ARG_CONFIG_PATH \
./run_train.sh \
--cfg_options \
fault_tolerance.enable=True \
fault_tolerance.replica_id=7 \
fault_tolerance.group_size=8 \
parallelism.data_parallel_replicate_degree=1