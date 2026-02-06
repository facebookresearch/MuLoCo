#!/bin/bash
#SBATCH -N 1
#SBATCH -p learn
#SBATCH --qos=parq_high
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=btherien@meta.com
#SBATCH --output=/home/btherien/logs/ddp/%j.out
#SBATCH --error=/home/btherien/logs/ddp/%j.err
#SBATCH --job-name=ddp_1

source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan

export WANDB_API_KEY=local-38f7f7a0bedbbafcbc6092cca0341b3f8be5dba0
wandb login --host https://fairwandb.org
export WANDB_PROJECT="diloco"

export NCCL_BUFFSIZE=8388608 
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_CROSS_NIC=1
export NCCL_NVLS_ENABLE=1
export TORCHDYNAMO_VERBOSE=1


NGPU=1 CONFIG_FILE="./config/ddp/1gpu_test.py" ./run_train.sh

