"""
[

1.0,
 0.707107,
 0.5,
 0.353553,
 0.25,
 0.176777,
 0.125,
 0.088388,
 0.0625,
 0.044194,
 0.03125,
 0.022097,
 0.015625,
 0.011049,
 0.007812,
 0.005524,
 0.003906,
 0.002762,
 0.001953,
 0.001381,
 0.000977,
 0.000691,
 0.000488,
 0.000345,
 0.000244,
 0.000173,
 0.000122,
 8.6e-05,
 6.1e-05,
 4.3e-05]
"""


from model_map import MODEL_STEP_MAP, LBS_MAP, USE_DP_PARALLELISM_MAP, OPTIMAL_WD_MAP
import numpy as np
import os
import time
OUTPUT_DIR = "jobs/muloco_paper/generated_job_files/ddp_h100_regular"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GPUS_PER_NODE = 8

def ddp_adamw(i_lr, alpha=0.0, suffix="''", sync_steps=32, topk=128, **kwargs):
    config_file = "config/ddp/adamw.py"
    return f"""optimizer.lr={i_lr} \
metrics.wandb_suffix={suffix}""", 32, config_file, 'adamw'



def ddp_muon(i_lr, alpha=0.0, suffix="''", nesterov=False, use_cautious_wd=False, use_polar_express=False, sync_steps=32, topk=128, **kwargs):
    config_file = "config/ddp/muon.py"
    return f"""optimizer.lr={i_lr} \
optimizer.nesterov={nesterov} \
optimizer.cautious_wd={use_cautious_wd} \
optimizer.use_polar_express={use_polar_express} \
optimizer.use_triton={not use_polar_express} \
metrics.wandb_suffix={suffix}""", 32, config_file, 'muon'



def get_eval_batches(nodes, replicas_per_node, full_global_batch_size=1024, max_local_batch_size=32):
    eval_global_batch_size = full_global_batch_size // (nodes * replicas_per_node)
    eval_local_batch_size = min(eval_global_batch_size, max_local_batch_size)
    return f"training.eval_global_batch_size={full_global_batch_size} training.eval_local_batch_size={eval_local_batch_size}"

def load_ckpt(ckpt):
    return f"checkpoint.initial_load_path={ckpt} \
checkpoint.enable_checkpoint=True"

def save_ckpt():
    return f"checkpoint.enable_checkpoint=True"


def save_pseudograds():
    return f"fault_tolerance.save_pseudograds=True \
fault_tolerance.pseudograd_path=/checkpoint/optim/btherien/pseudograds_muloco"

class WorkerConfiguration:


    def __init__(self, gpus_per_replica: int, nnodes: int, model_flavor: str):
        self.gpus_per_replica = gpus_per_replica
        self.nnodes = nnodes
        self.model_flavor = model_flavor

    def get_global_bs_per_worker(self, global_bs: int):
        return global_bs

    def get_parallelism_args(self):
        if USE_DP_PARALLELISM_MAP[self.model_flavor]:
            return f"""parallelism.data_parallel_replicate_degree={(GPUS_PER_NODE // self.gpus_per_replica) * self.nnodes}""" # 32 is needed due to extra mem overhead
        else:
            return f"""parallelism.data_parallel_replicate_degree=8 \
parallelism.data_parallel_shard_degree=16 \
parallelism.fsdp_reshard_after_forward='never'""" # 32 is needed due to extra mem overhead
#             return f"""parallelism.data_parallel_replicate_degree={(GPUS_PER_NODE // self.gpus_per_replica) * self.nnodes} \
# parallelism.data_parallel_shard_degree=1 \
# parallelism.fsdp_reshard_after_forward='never'""" # 32 is needed due to extra mem overhead

    def get_dp_world_size(self):
        return (GPUS_PER_NODE * self.nnodes) // self.gpus_per_replica

    def get_dp_max_local_bs(self, global_bs: int) -> int:
        assert global_bs % self.get_dp_world_size() == 0, "Global batch size must be divisible by the DP world size"
        return int(global_bs / self.get_dp_world_size())




      
import subprocess
import re

def get_running_nodes():
    """Get the number of nodes currently running for the user."""
    try:
        result = subprocess.run(['squeue','-q', 'h100_optim_high', '-u', os.environ.get('USER', 'btherien'), '-h'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if not lines or lines == ['']:
            return 0
        
        total_nodes = 0
        for line in lines:
            # Parse the NODES column (typically the 6th column in squeue output)
            parts = line.split()
            if len(parts) >= 6:
                try:
                    nodes = int(parts[6])
                    total_nodes += nodes
                except ValueError:
                    continue
        return total_nodes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0

def get_job_nodes(job_file):
    """Extract the number of nodes from a job script."""
    try:
        with open(job_file, 'r') as f:
            content = f.read()
        
        # Look for --nodes parameter in SBATCH directives
        match = re.search(r'#SBATCH\s+--nodes=(\d+)', content)
        if match:
            return int(match.group(1))
        return 1  # Default to 1 node if not specified
    except (FileNotFoundError, IOError):
        return 1

def launch_jobs_with_node_limit(max_nodes=14, ):
    """Launch jobs while respecting node limits."""
    # Sort files by job number extracted from filename
    def get_job_number(filename):
        match = re.search(r'_(\d+)\.sh$', filename)
        return int(match.group(1)) if match else 0
    
    # for filename in [x for x in sorted(os.listdir(OUTPUT_DIR), key=get_job_number) if get_job_number(x) > 6 ]:
    
    for filename in sorted(os.listdir(OUTPUT_DIR), key=get_job_number):
        print("launch_jobs_with_node_limit", filename)
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # Get nodes required for this job
        job_nodes = get_job_nodes(file_path)
        if job_nodes == 0:
            print("No nodes required for this job, skipping:", filename)
            continue
        
        # Wait until we have enough nodes available
        while True:
            running_nodes = get_running_nodes()
            if running_nodes + job_nodes <= max_nodes:
                break
            print(f"Waiting... Currently using {running_nodes} nodes, need {job_nodes} more (max: {max_nodes})")
            time.sleep(60)  # Wait 1 minutes before checking again
        
        print(f"Launching job {filename} requiring {job_nodes} nodes (current: {running_nodes})")
        os.system(f"sbatch {file_path}")
        time.sleep(0.1)  # Small delay to allow job to start



if __name__ == "__main__":
    print("in main")

    sync_steps = 30
    mainc = "outputs/j97hhisu_DP_Muon_LR0.04728708_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf"
    ckpts = [os.path.join(mainc, f"checkpoint/step-{step}/ft-replica-0") for step in [720, 1440, 3840, 5760]]
    ckpts = [""]

    
    # Remove all files in OUTPUT_DIR
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    count = 0
    topk = 128
    ckpt = ""
    alpha = 0.0

    use_nesterov_muon = False
    use_cautious_wd = False
    use_polar_express = False
    # suffix = f"nesterov_{use_nesterov_muon}_cwd_{use_cautious_wd}_pexp_{use_polar_express}"

    #     """
    # [

    # 1.0,
    #  0.707107,
    #  0.5,
    #  0.353553,
    #  0.25,
    #  0.176777,
    #  0.125,
    #  0.088388,
    #  0.0625,
    #  0.044194,
    #  0.03125,
    #  0.022097,
    #  0.015625,
    #  0.011049,
    #  0.007812,
    #  0.005524,
    #  0.003906,
    #  0.002762,
    #  0.001953,
    #  0.001381,
    #  0.000977,
    #  0.000691,
    #  0.000488,
    #  0.000345,
    #  0.000244,
    #  0.000173,
    #  0.000122,
    #  8.6e-05,
    #  6.1e-05,
    #  4.3e-05]
    # """
    # model_flavor = "llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf"
    # model_flavor = "llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf"
    # model_flavor = "llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf"
    model_flavor = "llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf"
    # model_flavor = 'llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf'

    SAVE_PSEUDOGRADS = False
    LOAD_CKPT = False

    rescaled_lr = 0.001381 * np.sqrt(2048/256)
    rescaled_b1 = 1 - (1 - 0.9) * 2048/256
    rescaled_b2 =  1 - (1 - 0.99) * 2048/256

    # suffix = '_b20.95'
    # suffix = '_sde_rescaled'
    # suffix = ''
    rescaled_b2 = 0.9875 # halflife 55
    # rescaled_b2 = 0.9847 # halflife 45
    # rescaled_b2 = 0.9804 # halflife 35
    rescaled_b2 = 0.9727 # halflife 25
    # rescaled_b2 = 0.9548 # halflife 15

    suffix = '_b2-'+str(rescaled_b2)
    suffix=''
    for (arg_func, warmup), lrs_to_sweep, wds_to_sweep, gbs_to_sweep in zip(
        [
            (ddp_muon, 500),
            # (ddp_adamw, 500),
        ],
        [   
            #     0.002762, 0.001953, 0.001381,
            [   

  0.044194,
     0.03125,
     0.022097, 
    #  0.001953,
    #  0.001381,
    #  0.000977,

            ],
            # [
            #     0.002762, 
            #     0.001953,
            #     0.001381,
            #  # For GBS 32 - need to check (below current best=0.0625)
            #     # 0.000977,
            # ], 
            # [
                # 0.002762,  # For GBS 2048 - need to check (above current best=0.015625)
            #     0.001953,
            #     0.001381,
            #     0.000977,
            # ], 
        ],
        [     #w512
            # [0.01,],
            # [0.001,],
            #w1024
            # [0.01,],
            # [0.01,],
            #w2048
            # [0.001,],
            # [0.001],
            #w2560
            [0.01],
        ],
        [
        #    [32],  # Muon batch sizes - fixed_data-parallel_Muon_B32_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf
        #    [2048]  # AdamW batch sizes - fixed_data-parallel_AdamW_B2048_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf
        # [512],
        # [1024,],



        [8192],
        # [256],


        ]

    ):
        # suffix = 'test'
        for ilr in lrs_to_sweep:
            # for alpha in alphas_to_sweep:
            for GLOBAL_BATCH_SIZE in gbs_to_sweep:

                STEPS = MODEL_STEP_MAP[model_flavor][GLOBAL_BATCH_SIZE]
                WD_BASE_STEPS = MODEL_STEP_MAP[model_flavor][512]
                # for topk in topks_to_sweep:

                for wd in wds_to_sweep:
                    for setting in [
                        WorkerConfiguration(gpus_per_replica=1, nnodes=16, model_flavor=model_flavor),
                    ]:


                        gbs = setting.get_global_bs_per_worker(GLOBAL_BATCH_SIZE)
                        nnodes = setting.nnodes
                        parallelism = setting.get_parallelism_args()



                        args, lbs, config_file, inner_optimizer_name  = arg_func(ilr, alpha, 
                        nesterov=use_nesterov_muon, 
                        sync_steps=sync_steps, 
                        suffix=suffix, 
                        topk=topk,
                        use_polar_express=use_polar_express,
                        use_cautious_wd=use_cautious_wd)

                        lbs = LBS_MAP[model_flavor] * 2

                        replicas_per_node = GPUS_PER_NODE // setting.gpus_per_replica
                        # args = periodic_centering_args(ilr, alpha, sync_steps=sync_steps)


                        if SAVE_PSEUDOGRADS:
                            pg_args = f" {save_pseudograds()}"
                        else:
                            pg_args = ""

                        if LOAD_CKPT:
                            ckpt_args = f" {load_ckpt(ckpt)}"
                        else:
                            ckpt_args = ""

                        # if setting == ddp_muon:
                        #     partition_args = f"""--partition h200 --account parq --qos parq_high"""
                        #     mult=2
                        # else:
                        partition_args = f"""--partition learn --account fair_amaia_cw_scale --qos scale"""
                        partition_args = f"""--partition h100 --account optim --qos optim_high"""
                        mult=1

                        print(setting.get_dp_max_local_bs(gbs))


                        print("WARNING: wd is not being swept")
                        wd = OPTIMAL_WD_MAP[model_flavor][inner_optimizer_name]

                        extra_args = f"""\
lr_scheduler.warmup_steps={warmup} \
model.flavor={model_flavor} \
metrics.eval_freq=15 \
metrics.wandb_project=dil_nemotroncc_v2 \
optimizer.weight_decay={wd*(WD_BASE_STEPS/STEPS)} \
training.global_batch_size={gbs} \
metrics.enable_lm_evaluation=False \
training.local_batch_size={min(setting.get_dp_max_local_bs(gbs), lbs*mult)} \
training.steps={STEPS+1} {parallelism} \
{args} \
{get_eval_batches(nnodes, replicas_per_node, max_local_batch_size=lbs)}{ckpt_args}{pg_args}"""

                        cmd = f"""python generate_job_file.py \
--config-file {config_file} \
--job-name DDP_lr{ilr}_gbs{gbs} \
{partition_args} \
--nodes {nnodes} \
--gpus-per-node {GPUS_PER_NODE} \
--mem 1000G \
--time 72:00:00 \
--use-ddp \
--mail-type BEGIN,END,FAIL \
--mail-user btherien@meta.com \
--output /home/btherien/logs/ddp/h100/%j.out \
--error /home/btherien/logs/ddp/h100/%j.err \
--extra-args "{extra_args}" \
--num-ft-replicas-per-node {replicas_per_node} \
--output-file {OUTPUT_DIR}/ddp_lr_sweep_{count}.sh"""

                        os.system(cmd)
                        count += 1

          

    launch_jobs_with_node_limit(max_nodes=500000)



