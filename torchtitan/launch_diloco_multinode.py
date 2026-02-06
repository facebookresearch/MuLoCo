import os


def launch_distributed_multinode(nodes, nodes_per_replica, extra_args, config_file, gpus_per_node=8, cpus_per_node=128, dryrun=False):
    """
    Launch a distributed training job where each replica spans multiple nodes.
    
    Args:
        nodes: List of node names
        nodes_per_replica: Number of nodes each replica spans
        extra_args: Extra arguments to pass to training command
        config_file: Path to config file
        gpus_per_node: Number of GPUs per node (default: 8)
        cpus_per_node: Number of CPUs per node (default: 128)
        dryrun: Print commands without executing them
    """
    NUM_NODES = len(nodes)
    GPUS_PER_NODE = gpus_per_node
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
    
    # CPU allocation: lighthouse gets 8 CPUs, torchrun gets the rest
    LIGHTHOUSE_CPUS = 8
    TORCHRUN_CPUS = cpus_per_node - LIGHTHOUSE_CPUS
    
    # Validate that nodes can be evenly divided into replicas
    assert NUM_NODES % nodes_per_replica == 0, \
        f"Number of nodes ({NUM_NODES}) must be divisible by nodes_per_replica ({nodes_per_replica})"
    
    NUM_REPLICAS = NUM_NODES // nodes_per_replica
    GPUS_PER_REPLICA = GPUS_PER_NODE * nodes_per_replica
    
    # Group nodes into chunks for each replica
    replica_node_groups = [
        nodes[i * nodes_per_replica : (i + 1) * nodes_per_replica]
        for i in range(NUM_REPLICAS)
    ]

    # Setup lighthouse on the first node
    lighthouse_command = f"""\
srun --nodes=1 \
--cpus-per-task={LIGHTHOUSE_CPUS} \
--ntasks=1 --overlap \
--cpu-bind=none \
-w "{nodes[0]}" \
torchft_lighthouse \
--min_replicas {NUM_REPLICAS} \
--quorum_tick_ms 100 \
--join_timeout_ms 10000 &"""

    print(lighthouse_command)

    if not dryrun:
        os.system(lighthouse_command)
    
    TORCHFT_LIGHTHOUSE = f"http://{nodes[0]}:29510"
    os.environ["TORCHFT_LIGHTHOUSE"] = TORCHFT_LIGHTHOUSE
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_dir = f"/home/btherien/logs/multinode_log/{SLURM_JOB_ID}"
    os.makedirs(log_dir, exist_ok=True)

    for replica_id, replica_nodes in enumerate(replica_node_groups):
        # First node of this replica group serves as rendezvous endpoint
        rdzv_node = replica_nodes[0]
        rdzv_port = 29500 + replica_id  # Unique port per replica
        node_list = ",".join(replica_nodes)
        
        # Determine if this is the last replica (for background/foreground execution)
        is_last_replica = (replica_id == NUM_REPLICAS - 1)
        log_suffix = f" 2>&1 | tee {log_dir}/log_{SLURM_JOB_ID}_{replica_id}.log"
        end = log_suffix if is_last_replica else log_suffix + " &"

        # Test command to verify environment
        test_cmd = f"""\
srun \
--nodes={nodes_per_replica} \
--ntasks={nodes_per_replica} \
--overlap \
--cpu-bind=none \
-w {node_list} bash -c 'echo $TORCHFT_LIGHTHOUSE > {log_dir}/test_{SLURM_JOB_ID}_{replica_id}_$(hostname).log'"""
        print(test_cmd, flush=True)
        if not dryrun:
            os.system(test_cmd)

        # Launch command for multi-node replica
        # Each node in the replica runs torchrun with proper rendezvous
        launch_command = f"""\
srun \
--nodes={nodes_per_replica} \
--ntasks={nodes_per_replica} \
--overlap \
--cpu-bind=none \
--cpus-per-task={TORCHRUN_CPUS} \
--export=ALL \
-w {node_list} \
bash -c 'export TORCHFT_LIGHTHOUSE={TORCHFT_LIGHTHOUSE}; \
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"; \
torchrun \
--nnodes={nodes_per_replica} \
--nproc_per_node={GPUS_PER_NODE} \
--rdzv_backend c10d \
--rdzv_endpoint={rdzv_node}:{rdzv_port} \
--rdzv_id {replica_id}_{SLURM_JOB_ID} \
--role rank \
--tee 3 \
-m torchtitan.train \
--job.config_file {config_file} \
--cfg_options fault_tolerance.enable=True \
fault_tolerance.replica_id={replica_id} \
fault_tolerance.group_size={NUM_REPLICAS} {extra_args}'""" + end
        # --local-ranks-filter 0 \
        print(launch_command, flush=True)

        if not dryrun:
            os.system(launch_command)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch distributed training job with multi-node replicas")
    parser.add_argument("--nodes", nargs="+", required=True, 
                       help="List of node names (e.g., node01 node02 node03 node04)")
    parser.add_argument("--nodes_per_replica", type=int, required=True,
                       help="Number of nodes each replica spans")
    parser.add_argument("--extra_args", type=str, default="",
                       help="Extra arguments to pass to training command")
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--gpus_per_node", type=int, default=8,
                       help="Number of GPUs per node (default: 8)")
    parser.add_argument("--cpus_per_node", type=int, default=128,
                       help="Number of CPUs per node (default: 128)")
    parser.add_argument("--dryrun", action="store_true",
                       help="Print commands without executing them")
    
    args = parser.parse_args()
    print(args.extra_args)

    # Handle quoted extra_args
    if args.extra_args and len(args.extra_args) >= 2:
        if args.extra_args[0] == '"' and args.extra_args[-1] == '"':
            extra_args = args.extra_args[1:-1]
        else:
            extra_args = args.extra_args
    else:
        extra_args = args.extra_args

    launch_distributed_multinode(
        args.nodes, 
        args.nodes_per_replica, 
        extra_args, 
        args.config_file, 
        args.gpus_per_node,
        args.cpus_per_node,
        args.dryrun
    )

