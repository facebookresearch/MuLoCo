import os
import time
OUTPUT_DIR = "jobs/muloco_paper/generated_job_files/diloco_no_compression_h100"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GPUS_PER_NODE = 8

from model_map import MODEL_STEP_MAP, LBS_MAP, USE_DP_PARALLELISM_MAP, OPTIMAL_WD_MAP

def diloco_args(i_lr, alpha=0.0, suffix="''", sync_steps=32, **kwargs):
    config_file = "config/diloco/2x_4gpu_diloco.py"
    return f"""fault_tolerance.use_gpa=False \
fault_tolerance.use_continuous_centering=False \
fault_tolerance.use_periodic_centering=False \
fault_tolerance.backup_device=cuda \
fault_tolerance.semi_sync_method=diloco \
fault_tolerance.sync_steps={sync_steps} \
checkpoint.exclude_from_loading=outer_optimizer_0, \
optimizer.lr={i_lr} \
fault_tolerance.fragment_update_alpha={alpha} \
metrics.wandb_suffix={suffix}""", 32, config_file, 'adamw'

def streaming_diloco_args(i_lr, alpha=0.0, suffix="_streaming_diloco", sync_steps=32, **kwargs):
    config_file = "config/diloco/2x_4gpu_streaming_diloco.py"
    return f"""fault_tolerance.use_gpa=False \
fault_tolerance.use_continuous_centering=False \
fault_tolerance.use_periodic_centering=False \
fault_tolerance.backup_device=cuda \
fault_tolerance.semi_sync_method=diloco \
fault_tolerance.sync_steps={sync_steps} \
checkpoint.exclude_from_loading=outer_optimizer_0, \
optimizer.lr={i_lr} \
fault_tolerance.fragment_update_alpha={alpha} \
metrics.wandb_suffix={suffix}""", 64, config_file, 'adamw'

def muloco_args(i_lr, alpha=0.0, suffix="''", sync_steps=32, **kwargs):
    config_file = "config/diloco/2x_4gpu_muloco_no-ckpt.py"
    return f"""fault_tolerance.use_gpa=False \
fault_tolerance.use_continuous_centering=False \
fault_tolerance.use_periodic_centering=False \
fault_tolerance.backup_device=cuda \
fault_tolerance.semi_sync_method=diloco \
fault_tolerance.sync_steps={sync_steps} \
checkpoint.exclude_from_loading=outer_optimizer_0, \
optimizer.lr={i_lr} \
fault_tolerance.fragment_update_alpha={alpha} \
metrics.wandb_suffix={suffix}""", 32, config_file, 'muon'



def muloco_muon_outer_args(i_lr, alpha=0.0, suffix="muon_outer", sync_steps=32, **kwargs):
    config_file = "config/diloco/2x_4gpu_muloco_outer_muon.py"
    return f"""fault_tolerance.use_gpa=False \
fault_tolerance.use_continuous_centering=False \
fault_tolerance.use_periodic_centering=False \
fault_tolerance.backup_device=cuda \
fault_tolerance.semi_sync_method=diloco \
fault_tolerance.sync_steps={sync_steps} \
checkpoint.exclude_from_loading=outer_optimizer_0, \
optimizer.lr={i_lr} \
fault_tolerance.fragment_update_alpha={alpha} \
metrics.wandb_suffix={suffix}""", 32, config_file, 'muon'








def load_ckpt(ckpt):
    return f"checkpoint.initial_load_path={ckpt} \
checkpoint.enable_checkpoint=True"


def save_pseudograds():
    return f"fault_tolerance.save_pseudograds=True \
fault_tolerance.pseudograd_path=/checkpoint/optim/btherien/pseudograds_muloco"

class WorkerConfiguration:
    ft_replicas_per_node: int
    nnodes: int


    def __init__(self, ft_replicas_per_node: int, nnodes: int, nodes_per_replica: int):
        self.ft_replicas_per_node = ft_replicas_per_node
        self.nnodes = nnodes
        self.nodes_per_replica = nodes_per_replica
        self.num_replicas = (self.ft_replicas_per_node * self.nnodes) // self.nodes_per_replica
        assert (GPUS_PER_NODE * self.nodes_per_replica) % self.ft_replicas_per_node == 0, "GPUS_PER_NODE must be divisible by ft_replicas_per_node"
        self.gpus_per_replica = (GPUS_PER_NODE * self.nodes_per_replica) // self.ft_replicas_per_node
        assert self.num_replicas % 1 == 0, "num_replicas must be an integer"


    def get_global_bs_per_worker(self, global_bs: int):
        assert global_bs % self.num_replicas == 0, f"global_bs {global_bs} must be divisible by self.num_replicas {self.num_replicas}"
        return global_bs // self.num_replicas

    def get_parallelism_args(self, model_flavor: str):
        if USE_DP_PARALLELISM_MAP[model_flavor]:
            return f"""parallelism.data_parallel_replicate_degree={self.gpus_per_replica}""" # 32 is needed due to extra mem overhead
        else:
#             return f"""parallelism.data_parallel_replicate_degree={self.gpus_per_replica} \
# parallelism.data_parallel_shard_degree=1 \
# parallelism.fsdp_reshard_after_forward='never'""" # 32 is needed due to extra mem overhead
#             return f"""parallelism.data_parallel_replicate_degree=1 \
# parallelism.data_parallel_shard_degree={self.gpus_per_replica} \
# parallelism.fsdp_reshard_after_forward='never'""" # 32 is needed due to extra mem overhead
           return f"""parallelism.data_parallel_replicate_degree=1 \
parallelism.data_parallel_shard_degree=8 \
parallelism.fsdp_reshard_after_forward='never'""" # 32 is needed due to extra mem overhead



    def get_replica_dp_world_size(self):
        return self.gpus_per_replica

    def get_replica_dp_max_local_bs(self, global_bs: int) -> int:
        assert global_bs % self.get_replica_dp_world_size() == 0, "Global batch size must be divisible by the DP world size"
        return int(global_bs / self.get_replica_dp_world_size())


    def get_eval_batches(self, full_global_batch_size=1024, max_local_batch_size=32):

        eval_global_batch_size = full_global_batch_size // self.num_replicas
        eval_local_batch_size_temp = eval_global_batch_size // self.gpus_per_replica
        eval_local_batch_size = min(eval_local_batch_size_temp, max_local_batch_size)
        return f"training.eval_global_batch_size={eval_global_batch_size} training.eval_local_batch_size={eval_local_batch_size}"

      
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

def launch_jobs_with_node_limit(max_nodes=14):
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

    sync_steps = 30
    ckpts = ["outputs/eaae9ewi_DP_AdamW_LR0.00948683_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf/checkpoint/step-3840/ft-replica-0",
    "outputs/eaae9ewi_DP_AdamW_LR0.00948683_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf/checkpoint/step-5760/ft-replica-0",
    "outputs/eaae9ewi_DP_AdamW_LR0.00948683_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf/checkpoint/step-1440/ft-replica-0",
    "outputs/eaae9ewi_DP_AdamW_LR0.00948683_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf/checkpoint/step-720/ft-replica-0",]
    # ckpt = "outputs/eaae9ewi_DP_AdamW_LR0.00948683_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf/checkpoint/step-720/ft-replica-0"
    mainc = "outputs/j97hhisu_DP_Muon_LR0.04728708_b10.9_b20.99_wd0.0001_steps7953_dps1_dpr8_cp1_tp1_pp1_B512_SL2048_llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf"
    ckpts = [os.path.join(mainc, f"checkpoint/step-{step}/ft-replica-0") for step in [720, 1440, 3840, 5760]]
    ckpts = [""]
    # model_flavor = "llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf"
    model_flavor = "llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf"
    model_flavor = 'llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf'
    model_flavor = "llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf"
    # model_flavor = 'llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf'

    SAVE_PSEUDOGRADS = False
    LOAD_CKPT = False

    # Remove all files in OUTPUT_DIR
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    count = 0
    topk = 128
    alpha = 0.0
    ckpt = ''
    sync_steps = 30

    SPARSELOCO_REPRODUCTION = False

#         1.0,
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
#  4.3e-05],

# 0.002762,0.001953,0.001381,0.000977,
    # USE_LR_06 = False
    for arg_func, worker_config_to_sweep in zip(
        [
            muloco_args,
            diloco_args,
        ],

        [ 
            # Muon configs
            [
(WorkerConfiguration(ft_replicas_per_node=1,  nnodes=16, nodes_per_replica=1), 30, [0.9], [1.0],   [0.015625], [2048]),

            ],
            # AdamW configs
            [
(WorkerConfiguration(ft_replicas_per_node=1,  nnodes=16, nodes_per_replica=1), 30, [0.9], [1.0],   [0.000691], [1024]),
            ],
        ],
    ):
        for setting, sync_steps, mom_to_sweep, olr_to_sweep, lrs_to_sweep, gbs_to_sweep in worker_config_to_sweep:
            for ilr in lrs_to_sweep:
                for GLOBAL_BATCH_SIZE in gbs_to_sweep:
                    STEPS = MODEL_STEP_MAP[model_flavor][GLOBAL_BATCH_SIZE]
                    WD_BASE_STEPS = MODEL_STEP_MAP[model_flavor][512]
                    for olr in olr_to_sweep:
                        for mom in mom_to_sweep:


                            gbs = setting.get_global_bs_per_worker(GLOBAL_BATCH_SIZE)
                            nnodes = setting.nnodes
                            parallelism = setting.get_parallelism_args(model_flavor)
                            ft_replicas_per_node = setting.ft_replicas_per_node
                            assert GPUS_PER_NODE % ft_replicas_per_node == 0, "GPUS_PER_NODE must be divisible by ft_replicas_per_node"

                            args, lbs, config_file, inner_optimizer_name = arg_func(ilr, alpha, sync_steps=sync_steps,topk=topk,suffix='')
                            lbs = LBS_MAP[model_flavor] * 2
                            # args = periodic_centering_args(ilr, alpha, sync_steps=sync_steps)

                            k = setting.num_replicas


                            if SAVE_PSEUDOGRADS:
                                pg_args = f" {save_pseudograds()}"
                            else:
                                pg_args = ""

                            if LOAD_CKPT:
                                ckpt_args = f" {load_ckpt(ckpt)}"
                            else:
                                ckpt_args = ""
                            
                            print(setting.get_replica_dp_max_local_bs(gbs))

                            wd = OPTIMAL_WD_MAP[model_flavor][inner_optimizer_name]
                            partition_args = f"""--partition learn --account fair_amaia_cw_scale --qos scale"""
                            partition_args = f"""--partition h100 --account optim --qos optim_high"""


                            extra_args = f"""\
lr_scheduler.warmup_steps=500 \
model.flavor={model_flavor} \
optimizer.weight_decay={wd*(WD_BASE_STEPS/(STEPS*k))} \
outer_optimizer.kwargs.lr={olr} \
outer_optimizer.kwargs.momentum={mom} \
metrics.eval_freq=15 \
training.gc_freq=250 \
metrics.enable_lm_evaluation=False \
metrics.wandb_project=dil_nemotroncc_v2 \
checkpoint.enable_checkpoint=True \
training.global_batch_size={gbs} \
training.local_batch_size={min(setting.get_replica_dp_max_local_bs(gbs), lbs)} \
training.steps={STEPS+1} {parallelism} \
{args} \
{setting.get_eval_batches(max_local_batch_size=lbs)}{ckpt_args}{pg_args}"""

                            cmd = f"""python generate_job_file.py \
--config-file {config_file} \
--job-name k{setting.num_replicas}_lr{ilr}_gbs{gbs} \
{partition_args} \
--nodes {nnodes} \
--gpus-per-node {GPUS_PER_NODE} \
--nodes-per-replica {setting.nodes_per_replica} \
--time 48:00:00 \
--mail-type BEGIN,END,FAIL \
--mail-user btherien@meta.com \
--output /home/btherien/logs/diloco/h100/%j.out \
--error /home/btherien/logs/diloco/h100/%j.err \
--extra-args "{extra_args}" \
--num-ft-replicas-per-node {ft_replicas_per_node} \
--output-file {OUTPUT_DIR}/diloco_inner_lr_sweep_{count}.sh"""

                            os.system(cmd)
                            count += 1

          

    launch_jobs_with_node_limit(max_nodes=500)


