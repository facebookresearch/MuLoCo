import os


def launch_distributed(nodes, num_ft_replicas_per_node, extra_args, config_file, gpus_per_node=8, cpus_per_node=128, dryrun=False):
    """
    Launch a distributed training job.
    """
    NUM_NODES = len(nodes)
    NUM_REPLICAS = NUM_NODES * num_ft_replicas_per_node
    GPUS_PER_NODE = gpus_per_node
    GPUS_PER_REPLICA = int(GPUS_PER_NODE / num_ft_replicas_per_node)
    GPUS = [str(x) for x in range(GPUS_PER_NODE)]
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
    
    # CPU allocation: lighthouse gets 8 CPUs, rest split among replicas
    LIGHTHOUSE_CPUS = 8
    CPUS_PER_REPLICA = (cpus_per_node - LIGHTHOUSE_CPUS) // num_ft_replicas_per_node


    # setup lighthouse

    lighthouse_command = f"""\
srun --nodes=1 \
--ntasks=1 --overlap \
--cpu-bind=none \
--cpus-per-task={LIGHTHOUSE_CPUS} \
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

    assert GPUS_PER_NODE % num_ft_replicas_per_node == 0

    replica_id = 0
    for n,node_name in enumerate(nodes):
        for r in range(num_ft_replicas_per_node):
            gpus = GPUS[r*GPUS_PER_REPLICA:(r+1)*GPUS_PER_REPLICA]
            # print(gpus)
            first =  f" > {log_dir}/log_{SLURM_JOB_ID}_{replica_id}.log 2>&1"
            end = first if n == (NUM_NODES - 1) and r == (num_ft_replicas_per_node - 1) \
                else first + " &"

            # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
            test_cmd = f"""\
CUDA_VISIBLE_DEVICES={",".join(gpus)} srun \
--export=ALL,CUDA_VISIBLE_DEVICES={",".join(gpus)} \
--nodes=1 \
--ntasks=1 \
--overlap \
--cpu-bind=none \
-w {node_name} bash -c 'export CUDA_VISIBLE_DEVICES={",".join(gpus)}; echo $TORCHFT_LIGHTHOUSE > {log_dir}/test_{SLURM_JOB_ID}_{replica_id}.log'"""
            print(test_cmd, flush=True)
            if not dryrun:
                os.system(test_cmd)

            launch_command = f"""\
srun \
--nodes=1 \
--ntasks=1 \
--overlap \
--cpu-bind=none \
--cpus-per-task={CPUS_PER_REPLICA} \
-w {node_name} \
bash -c 'export CUDA_VISIBLE_DEVICES={",".join(gpus)}; \
export TORCHFT_LIGHTHOUSE={TORCHFT_LIGHTHOUSE}; \
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"; \
torchrun \
--nproc_per_node={GPUS_PER_REPLICA} \
--rdzv_backend c10d \
--rdzv_endpoint=localhost:0 \
--role rank \
--tee 3 \
-m torchtitan.train \
--job.config_file {config_file} \
--cfg_options fault_tolerance.enable=True \
fault_tolerance.replica_id={replica_id} \
fault_tolerance.group_size={NUM_REPLICAS} {extra_args}'""" + end



            print(launch_command, flush=True)
            replica_id += 1

            if not dryrun:
                os.system(launch_command)

# --local-ranks-filter 0 \



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch distributed training job")
    parser.add_argument("--nodes", nargs="+", required=True, 
                       help="List of node names (e.g., node01 node02)")
    parser.add_argument("--num_ft_replicas_per_node", type=int, required=True,
                       help="Number of fault tolerance replicas per node")
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

    # assert args.extra_args[0] == '"' and args.extra_args[-1] == '"'
    if args.extra_args[0] == '"' and args.extra_args[-1] == '"':
        extra_args = args.extra_args[1:-1]
    else:
        extra_args = args.extra_args

    launch_distributed(args.nodes, args.num_ft_replicas_per_node, extra_args, 
                      args.config_file, args.gpus_per_node, args.cpus_per_node, args.dryrun)





