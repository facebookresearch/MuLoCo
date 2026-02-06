
import argparse


SCALE = {
    "account": "fair_amaia_cw_scale",
    "partition": "learn",
    "qos": "scale",
    "mem": "1000G",
}

EXPLORE = {
    "account": "fair_amaia_cw_explore",
    "partition": "learn",
    "qos": "explore",
    "mem": "1000G",
}

LOWEST = {
    "account": "fair_amaia_cw_explore",
    "partition": "learn",
    "qos": "lowest",
    "mem": "1000G",
}

PARQ_HIGH = {
    "account": "parq",
    "partition": "learn",
    "qos": "h200_parq_high",
    "mem": "1900G",
}

H200_ALIGNMENT_SHARED = {
    "account": "optim",
    "partition": "learn",
    "qos": "h200_alignment_shared",
    "mem": "1900G",
}

H100_ALIGNMENT_SHARED = {
    "account": "optim",
    "partition": "learn",
    "qos": "h100_alignment_shared",
    "mem": "1000G",
}

OPTIM_HIGH = {
    "account": "optim",
    "partition": "learn",
    "qos": "h100_optim_high",
    "mem": "1000G",
}

QOS_DEFAULT_MAP = {
    "lowest": LOWEST,
    "alignment_shared_h100": H100_ALIGNMENT_SHARED,
    "alignment_shared_h200": H200_ALIGNMENT_SHARED,
    "parq_high": PARQ_HIGH,
    "optim_high": OPTIM_HIGH,
    "explore": EXPLORE,
    "scale": SCALE,
}


QOS_CPUS_PER_NODE = {
    "lowest": 128,
    "explore": 128,
    "scale": 128,
    "alignment_shared_h100": 192,
    "alignment_shared_h200": 192,
    "parq_high": 192,
    "optim_high": 192,
}


MEM_PER_NODE = {
    "lowest": "1600G",
    "explore": "1600G",
    "scale": "1600G",
    "alignment_shared_h100": "1900G",
    "alignment_shared_h200": "1900G",
    "parq_high": "1900G",
    "optim_high": "1900G",
}




SETUP_PREFIX_CW = """source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan
export WANDB_PROJECT="dil_nemotroncc_v2"


LOG_PREFIX="[DILOCO]"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


echo "$LOG_PREFIX Nodes: ${nodes[@]}"
echo "$LOG_PREFIX Nodes array: ${nodes_array[@]}"
echo "$LOG_PREFIX Running: srun --nodes=1 --ntasks=1 -w \"$head_node\" hostname --ip-address" >&2
echo $LOG_PREFIX Node IP: $head_node_ip
export LOGLEVEL=INFO
export RUST_BACKTRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CUMEM_ENABLE=1
"""

SETUP_PREFIX_AWS = """source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan
LOG_PREFIX="[DILOCO]"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


echo "$LOG_PREFIX Nodes: ${nodes[@]}"
echo "$LOG_PREFIX Nodes array: ${nodes_array[@]}"
echo "$LOG_PREFIX Running: srun --nodes=1 --ntasks=1 -w \"$head_node\" hostname --ip-address" >&2
echo $LOG_PREFIX Node IP: $head_node_ip
export LOGLEVEL=INFO
export RUST_BACKTRACE=1
export WANDB_PROJECT="dil_nemotroncc_v2"

# Enable for A100/H100
export FI_PROVIDER="efa"
export NCCL_IB_DISABLE=1

# EFA optimizations for AWS
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_TX_SIZE=8192
export FI_EFA_RX_SIZE=8192

# debugging flags (disabled for performance)
export NCCL_DEBUG=OFF
export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# Network configuration
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=8388608
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

# NCCL optimizations
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
# Note: NCCL_ALGO=Tree removed - doesn't support AllGather with int8

# H100 NVLink optimization
export NCCL_NVLS_ENABLE=1

# CUDA optimization for overlapping compute/communication
export CUDA_DEVICE_MAX_CONNECTIONS=1

export RUST_BACKTRACE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
"""


def make_slurm_header(
    job_name="diloco",
    partition="learn", 
    account="fair_amaia_cw_explore", 
    qos="lowest",
    nodes=2,
    gpus_per_node=8,
    ntasks_per_node=9, 
    cpus_per_task=14, 
    mem="1000G", 
    time="36:00:00", 
    mail_type="BEGIN,END,FAIL", 
    mail_user="btherien@meta.com", 
    output="/home/btherien/logs/diloco/%j.out", 
    error="/home/btherien/logs/diloco/%j.err"):
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH -A {account}
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user={mail_user}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --requeue"""


def make_launch(extra_args, config_file, gpus_per_node=8, num_ft_replicas_per_node=8, cpus_per_node=128):
    return f"""python launch_distributed.py \\
    --nodes "${{nodes_array[@]}}" \\
    --num_ft_replicas_per_node {num_ft_replicas_per_node} \\
    --config_file {config_file} \\
    --gpus_per_node {gpus_per_node} \\
    --cpus_per_node {cpus_per_node} \\
    --extra_args \"{extra_args}\""""


def make_launch_ddp(extra_args, config_file, gpus_per_node=8, nnodes=4, cpus_per_node=128):
    return f"""srun --cpus-per-task={cpus_per_node} torchrun \
--nnodes {nnodes} \
--nproc_per_node {gpus_per_node} \
--rdzv_id 101 \
--rdzv_backend c10d \
--rdzv_endpoint "${{head_node_ip}}:29500" \
./torchtitan/train.py --job.config_file {config_file} --cfg_options {extra_args}"""


def make_launch_multinode(extra_args, config_file, gpus_per_node=8, nodes_per_replica=2, cpus_per_node=128):
    """Generate launch command for multi-node replicas (workers spanning multiple nodes)."""
    return f"""python launch_diloco_multinode.py \\
    --nodes "${{nodes_array[@]}}" \\
    --nodes_per_replica {nodes_per_replica} \\
    --config_file {config_file} \\
    --gpus_per_node {gpus_per_node} \\
    --cpus_per_node {cpus_per_node} \\
    --extra_args \"{extra_args}\""""


SUFFIX = """
# wait
exit 0
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM job files for DiLoCo training")
    
    # Arguments for make_slurm_header function
    parser.add_argument("--job-name", default="diloco", help="SLURM job name")
    parser.add_argument("--partition", default="learn", help="SLURM partition")
    parser.add_argument("--account", default="fair_amaia_cw_explore", help="SLURM account")
    parser.add_argument("--qos", default="lowest", choices=list(QOS_DEFAULT_MAP.keys()), help="SLURM quality of service")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="Number of GPUs per node")
    parser.add_argument("--ntasks-per-node", type=int, default=None, help="Number of tasks per node (auto-computed if not specified)")
    parser.add_argument("--cpus-per-node", type=int, default=None, help="Number of CPUs per node (auto-computed from QOS if not specified)")
    parser.add_argument("--mem", default="1000G", help="Memory allocation")
    parser.add_argument("--time", default="36:00:00", help="Time limit")
    parser.add_argument("--mail-type", default="BEGIN,END,FAIL", help="Mail notification types")
    parser.add_argument("--mail-user", default="btherien@meta.com", help="Email for notifications")
    parser.add_argument("--output", default="/home/btherien/logs/diloco/%j.out", help="Output file path")
    parser.add_argument("--error", default="/home/btherien/logs/diloco/%j.err", help="Error file path")
    
    # Arguments for make_launch function
    parser.add_argument("--extra-args", default="", help="Extra arguments for launch_distributed.py")
    parser.add_argument("--config-file", required=True, help="Configuration file path")
    parser.add_argument("--num-ft-replicas-per-node", type=int, default=8, help="Number of FT replicas per node")
    parser.add_argument("--nodes-per-replica", type=int, default=1, help="Number of nodes each replica spans (use >1 for multi-node workers)")
    parser.add_argument("--use-ddp", action="store_true")
    
    # Output file argument
    parser.add_argument("--output-file", help="Output file path for generated job script")
    
    args = parser.parse_args()
    
    # Compute cpus_per_node from QOS if not specified
    cpus_per_node = args.cpus_per_node if args.cpus_per_node is not None else QOS_CPUS_PER_NODE[args.qos]
    
    # Compute ntasks_per_node based on job type if not specified
    if args.ntasks_per_node is not None:
        ntasks_per_node = args.ntasks_per_node
    elif args.use_ddp:
        # DDP: 1 task per node, uses all CPUs
        ntasks_per_node = 1
    elif args.nodes_per_replica > 1:
        # Multinode DiLoCo: 2 tasks per node (lighthouse + torchrun)
        ntasks_per_node = 2
    else:
        # Regular DiLoCo: 1 (lighthouse) + num_ft_replicas_per_node tasks
        ntasks_per_node = 1 + args.num_ft_replicas_per_node
    
    # Compute cpus_per_task for SLURM header based on job type
    # This ensures we request all available CPUs on the node
    # Individual srun commands will specify their exact CPU needs
    cpus_per_task_header = cpus_per_node // ntasks_per_node

    cpus_per_node = cpus_per_task_header * ntasks_per_node
    
    # Generate SLURM header
    slurm_header = make_slurm_header(
        job_name=args.job_name,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task_header,
        # mem=MEM_PER_NODE[args.qos],
        time=args.time,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        output=args.output,
        error=args.error,
        **QOS_DEFAULT_MAP[args.qos]
    )
    
    # Choose setup prefix based on type
    if args.qos in ["lowest", "explore", "scale"]:
        setup_prefix = SETUP_PREFIX_CW
    else:
        setup_prefix = SETUP_PREFIX_AWS
    
    # Generate launch command
    if args.use_ddp:
        launch_command = make_launch_ddp(
            extra_args=args.extra_args,
            config_file=args.config_file,
            gpus_per_node=args.gpus_per_node,
            nnodes=args.nodes,
            cpus_per_node=cpus_per_node
        )
    elif args.nodes_per_replica > 1:
        # Use multi-node launcher when replicas span multiple nodes
        launch_command = make_launch_multinode(
            extra_args=args.extra_args,
            config_file=args.config_file,
            gpus_per_node=args.gpus_per_node,
            nodes_per_replica=args.nodes_per_replica,
            cpus_per_node=cpus_per_node
        )
    else:
        launch_command = make_launch(
            extra_args=args.extra_args,
            config_file=args.config_file,
            gpus_per_node=args.gpus_per_node,
            num_ft_replicas_per_node=args.num_ft_replicas_per_node,
            cpus_per_node=cpus_per_node
        )
        
    # Combine all parts
    job_script = f"{slurm_header}\n\n{setup_prefix}\n\n{launch_command}\n\n{SUFFIX}\n"
    
    # Output the job script
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(job_script)
        print(f"Job script written to: {args.output_file}")
    else:
        print(job_script)



"""
# Single-node workers (default, each replica on one node or fraction of a node)
python generate_job_file.py \
    --config-file /path/to/your/config.py \
    --job-name diloco \
    --partition learn \
    --account fair_amaia_cw_explore \
    --qos lowest \
    --nodes 2 \
    --gpus-per-node 8 \
    --ntasks-per-node 9 \
    --cpus-per-task 8 \
    --mem 1000G \
    --time 36:00:00 \
    --mail-type BEGIN,END,FAIL \
    --mail-user btherien@meta.com \
    --output /home/btherien/logs/diloco/%j.out \
    --error /home/btherien/logs/diloco/%j.err \
    --extra-args "" \
    --num-ft-replicas-per-node 8

# Multi-node workers (each replica spans multiple nodes)
python generate_job_file.py \
    --config-file /path/to/your/config.py \
    --job-name diloco_multinode \
    --qos optim_high \
    --nodes 4 \
    --gpus-per-node 8 \
    --nodes-per-replica 2 \
    --output-file my_job.sh
"""