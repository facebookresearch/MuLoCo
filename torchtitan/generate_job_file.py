
import argparse


# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================
# Define your SLURM cluster's QOS/partition settings below.
# Each entry should specify account, partition, qos, and memory per node.
#
# Example:
#   MY_QOS = {
#       "account": "my-slurm-account",
#       "partition": "gpu",
#       "qos": "normal",
#       "mem": "500G",
#   }

EXAMPLE_QOS = {
    "account": "my-account",
    "partition": "gpu",
    "qos": "normal",
    "mem": "500G",
}

# Map QOS names to their configurations.
# The keys here are used with --qos on the command line.
QOS_DEFAULT_MAP = {
    "normal": EXAMPLE_QOS,
}

# CPUs available per node for each QOS.
QOS_CPUS_PER_NODE = {
    "normal": 128,
}

# Memory per node for each QOS.
MEM_PER_NODE = {
    "normal": "500G",
}


# ============================================================================
# SETUP PREFIX
# ============================================================================
# Shell commands run at the start of each job to set up the environment.
# Customize for your cluster: module loads, NCCL settings, proxy config, etc.

SETUP_PREFIX = """source $MULOCO_PATH/setup.sh
cd $MULOCO_PATH/torchtitan

LOG_PREFIX="[DILOCO]"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "$LOG_PREFIX Nodes: ${nodes[@]}"
echo "$LOG_PREFIX Nodes array: ${nodes_array[@]}"
echo "$LOG_PREFIX Running: srun --nodes=1 --ntasks=1 -w \\"$head_node\\" hostname --ip-address" >&2
echo $LOG_PREFIX Node IP: $head_node_ip
export LOGLEVEL=INFO
export RUST_BACKTRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add cluster-specific NCCL/network settings here, e.g.:
# export NCCL_BUFFSIZE=8388608
# export NCCL_CROSS_NIC=1
# export NCCL_NVLS_ENABLE=1
# export NCCL_CUMEM_ENABLE=1
"""


# ============================================================================
# SLURM HEADER
# ============================================================================

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
    mail_user="user@example.com",
    output="logs/%j.out",
    error="logs/%j.err"):
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


# ============================================================================
# LAUNCH COMMANDS
# ============================================================================

def make_launch(extra_args, config_file, gpus_per_node=8, num_ft_replicas_per_node=8, cpus_per_node=128):
    """Generate launch command for single-node replicas (DiLoCo/MuLoCo via TorchFT)."""
    return f"""python launch_distributed.py \\
    --nodes "${{nodes_array[@]}}" \\
    --num_ft_replicas_per_node {num_ft_replicas_per_node} \\
    --config_file {config_file} \\
    --gpus_per_node {gpus_per_node} \\
    --cpus_per_node {cpus_per_node} \\
    --extra_args \"{extra_args}\""""


def make_launch_ddp(extra_args, config_file, gpus_per_node=8, nnodes=4, cpus_per_node=128):
    """Generate launch command for DDP training (torchrun + srun)."""
    return f"""srun --cpus-per-task={cpus_per_node} torchrun \\
--nnodes {nnodes} \\
--nproc_per_node {gpus_per_node} \\
--rdzv_id 101 \\
--rdzv_backend c10d \\
--rdzv_endpoint "${{head_node_ip}}:29500" \\
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM job files for DiLoCo/MuLoCo training")

    # Arguments for make_slurm_header function
    parser.add_argument("--job-name", default="diloco", help="SLURM job name")
    parser.add_argument("--partition", default="gpu", help="SLURM partition")
    parser.add_argument("--account", default="my-account", help="SLURM account")
    parser.add_argument("--qos", default="normal", choices=list(QOS_DEFAULT_MAP.keys()), help="SLURM quality of service")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="Number of GPUs per node")
    parser.add_argument("--ntasks-per-node", type=int, default=None, help="Number of tasks per node (auto-computed if not specified)")
    parser.add_argument("--cpus-per-node", type=int, default=None, help="Number of CPUs per node (auto-computed from QOS if not specified)")
    parser.add_argument("--mem", default="500G", help="Memory allocation")
    parser.add_argument("--time", default="36:00:00", help="Time limit")
    parser.add_argument("--mail-type", default="BEGIN,END,FAIL", help="Mail notification types")
    parser.add_argument("--mail-user", default="user@example.com", help="Email for notifications")
    parser.add_argument("--output", default="logs/%j.out", help="Output file path")
    parser.add_argument("--error", default="logs/%j.err", help="Error file path")

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
        time=args.time,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        output=args.output,
        error=args.error,
        **QOS_DEFAULT_MAP[args.qos]
    )

    # Use the setup prefix
    setup_prefix = SETUP_PREFIX

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
# Example: DDP baseline
python generate_job_file.py \\
    --config-file config/ddp/muon.py \\
    --job-name ddp_muon \\
    --qos normal \\
    --nodes 4 \\
    --gpus-per-node 8 \\
    --time 24:00:00 \\
    --use-ddp \\
    --extra-args "optimizer.lr=0.03125 model.flavor=llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf training.global_batch_size=512 training.local_batch_size=16 training.steps=8160" \\
    --output-file jobs/my_ddp_job.sh

# Example: DiLoCo with single-node replicas (8 replicas per node, 2 nodes = 16 workers)
python generate_job_file.py \\
    --config-file config/diloco/2x_4gpu_diloco.py \\
    --job-name diloco_k16 \\
    --qos normal \\
    --nodes 2 \\
    --gpus-per-node 8 \\
    --num-ft-replicas-per-node 8 \\
    --time 48:00:00 \\
    --extra-args "fault_tolerance.sync_steps=30 optimizer.lr=0.003906 training.global_batch_size=64 training.local_batch_size=64 training.steps=8160" \\
    --output-file jobs/my_diloco_job.sh

# Example: MuLoCo with multi-node replicas (4 nodes, 2 per replica = 2 replicas)
python generate_job_file.py \\
    --config-file config/diloco/2x_4gpu_muloco.py \\
    --job-name muloco_multinode \\
    --qos normal \\
    --nodes 4 \\
    --gpus-per-node 8 \\
    --nodes-per-replica 2 \\
    --time 48:00:00 \\
    --extra-args "fault_tolerance.sync_steps=30 optimizer.lr=0.015625 training.global_batch_size=1024 training.local_batch_size=8 training.steps=4080" \\
    --output-file jobs/my_muloco_multinode_job.sh
"""
