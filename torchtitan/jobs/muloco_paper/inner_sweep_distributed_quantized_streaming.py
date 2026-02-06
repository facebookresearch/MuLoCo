import os
import time
OUTPUT_DIR = "jobs/muloco_paper/generated_job_files/quantized_streaming"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GPUS_PER_NODE = 8



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
metrics.wandb_suffix={suffix}""", 64, config_file



def streaming_muloco_args(i_lr, alpha=0.0, suffix="_streaming_muloco", sync_steps=32, **kwargs):
    config_file = "config/diloco/2x_4gpu_streaming_muloco.py"
    return f"""fault_tolerance.use_gpa=False \
fault_tolerance.use_continuous_centering=False \
fault_tolerance.use_periodic_centering=False \
fault_tolerance.backup_device=cuda \
fault_tolerance.semi_sync_method=diloco \
fault_tolerance.sync_steps={sync_steps} \
checkpoint.exclude_from_loading=outer_optimizer_0, \
optimizer.lr={i_lr} \
fault_tolerance.fragment_update_alpha={alpha} \
metrics.wandb_suffix={suffix}""", 32, config_file





def get_eval_batches(nodes, replicas_per_node, full_global_batch_size=1024, max_local_batch_size=32):
    eval_global_batch_size = full_global_batch_size // (nodes * replicas_per_node)
    eval_local_batch_size = min(eval_global_batch_size, max_local_batch_size)
    return f"training.eval_global_batch_size={eval_global_batch_size} training.eval_local_batch_size={eval_local_batch_size}"

def load_ckpt(ckpt):
    return f"checkpoint.initial_load_path={ckpt} \
checkpoint.enable_checkpoint=True"


def save_pseudograds():
    return f"fault_tolerance.save_pseudograds=True \
fault_tolerance.pseudograd_path=/checkpoint/optim/btherien/pseudograds_muloco"

class WorkerConfiguration:
    ft_replicas_per_node: int
    nnodes: int


    def __init__(self, ft_replicas_per_node: int, nnodes: int):
        self.ft_replicas_per_node = ft_replicas_per_node
        self.nnodes = nnodes

    def get_global_bs_per_worker(self, global_bs: int):
        denom = self.ft_replicas_per_node * self.nnodes
        assert global_bs % denom == 0, f"global_bs {global_bs} must be divisible by ft_replicas_per_node {self.ft_replicas_per_node} * nnodes {self.nnodes} = {denom}"
        return global_bs // denom

    def get_parallelism_args(self):
        return f"""parallelism.data_parallel_replicate_degree={GPUS_PER_NODE // self.ft_replicas_per_node}""" # 32 is needed due to extra mem overhead




      
import subprocess
import re

def get_running_nodes():
    """Get the number of nodes currently running for the user."""
    try:
        result = subprocess.run(['squeue','-q', 'h200_parq_high', '-u', os.environ.get('USER', 'btherien'), '-h'], 
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
        time.sleep(5)  # Small delay to allow job to start



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
    model_flavor = "llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf"
    STEPS = 7953 #2445
    GLOBAL_BATCH_SIZE = 2048
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
    sync_steps=30



    # USE_LR_06 = False

    for arg_func, lrs_to_sweep, olr_to_sweep, worker_config_to_sweep, quantization_bins in zip(
        [
            # continuous_centering_args, 
            # periodic_centering_args, 
            # muloco_args, 
            # diloco_args, 
            # muloco_muon_outer_args,

            muloco_qunatized_outer_args,
            streaming_muloco_args,
            # diloco_quantized_outer_args,
            # gpa_args,

            # sparseloco_diloco_args,
            # streaming_sparseloco_diloco_args,
            # diloco_args,
            # streaming_diloco_args,
            # sparseloco_muloco_args, 
        ],
        # inner_optimizer.kwargs.lr
        [     
            [0.04728708, 0.02236068,0.01057371], 
            [0.003, 0.0016870, 0.00094868], 
        ],
        # outer_optimizer.kwargs.lr
        [    
            # [0.0],
            # [0.1, 0.04728708, 0.02236068, 0.01057371,0.005, 0.003, 0.0016870, 0.00094868], 
            # [1.0,0.9,0.8,0.7,],
            [0.9,0.8],
            
            #0.5,0.8,0.9,1.0,1.1,1.2,1.5,2.0,]
            # [0.0]
        ],
        # workers, BS and sync_steps
        [   [
            (WorkerConfiguration(ft_replicas_per_node=8,  nnodes=1,), 512, 30),
            ], 
            # [
            # (WorkerConfiguration(ft_replicas_per_node=8,  nnodes=1,), 512, 30),
            # ], 
        ],
        # bins and error_decay
        [
            # [0], 
            # [0],
            # [16, 256],
            # [16, 256],
            # [(0.95,4),(0.8,4),(0.7,4),(0.5,4)],
            # [4,16,256],
            # [4,16,256],
            [(0.9,4),],
            [(0.9,16),],
            [(0.9,256),],
        ]
    ):
        for ilr in lrs_to_sweep:
            # for alpha in alphas_to_sweep:
            for error_decay, q in quantization_bins:
                for olr in olr_to_sweep:
                # for ckpt in ckpts:
                    for setting, GLOBAL_BATCH_SIZE, sync_steps in worker_config_to_sweep:

                        gbs = setting.get_global_bs_per_worker(GLOBAL_BATCH_SIZE)
                        nnodes = setting.nnodes
                        parallelism = setting.get_parallelism_args()
                        ft_replicas_per_node = setting.ft_replicas_per_node
                        assert GPUS_PER_NODE % ft_replicas_per_node == 0, "GPUS_PER_NODE must be divisible by ft_replicas_per_node"

                        args, lbs, config_file = arg_func(ilr, alpha, sync_steps=sync_steps,topk=topk)
                        # args = periodic_centering_args(ilr, alpha, sync_steps=sync_steps)

                        k = ft_replicas_per_node * nnodes


                        if SAVE_PSEUDOGRADS:
                            pg_args = f" {save_pseudograds()}"
                        else:
                            pg_args = ""

                        if LOAD_CKPT:
                            ckpt_args = f" {load_ckpt(ckpt)}"
                        else:
                            ckpt_args = ""
                        

                        # if USE_LR_06:
                        #     args += " outer_optimizer.kwargs.lr=0.6"

                        extra_args = f"""\
model.flavor={model_flavor} \
metrics.eval_freq={sync_steps} \
outer_optimizer.kwargs.lr={olr} \
checkpoint.enable_checkpoint=False \
outer_optimizer.kwargs.quantization_bins={q} \
outer_optimizer.kwargs.error_decay={error_decay} \
outer_optimizer.kwargs.skip_norm_quantization=True \
outer_optimizer.kwargs.simulate_quantization_after_reduce=True \
outer_optimizer.kwargs.use_ef=True \
outer_optimizer.kwargs.compressor_type=row_wise_statistical \
outer_optimizer.kwargs.skip_embedding_quantization=True \
training.global_batch_size={gbs} \
training.local_batch_size={min(gbs, lbs)} \
training.steps={STEPS} {parallelism} \
{args} \
{get_eval_batches(nnodes, ft_replicas_per_node, max_local_batch_size=lbs)}{ckpt_args}{pg_args}"""

                        cmd = f"""python generate_job_file.py \
--config-file {config_file} \
--job-name k{ft_replicas_per_node*nnodes}_lr{ilr}_gbs{gbs} \
--partition learn \
--account parq \
--qos parq_high \
--nodes {nnodes} \
--gpus-per-node {GPUS_PER_NODE} \
--ntasks-per-node 9 \
--cpus-per-task 8 \
--mem 1000G \
--time 36:00:00 \
--mail-type BEGIN,END,FAIL \
--mail-user btherien@meta.com \
--output /home/btherien/logs/diloco/h200/%j.out \
--error /home/btherien/logs/diloco/h200/%j.err \
--extra-args "{extra_args}" \
--num-ft-replicas-per-node {ft_replicas_per_node} \
--output-file {OUTPUT_DIR}/diloco_inner_lr_sweep_{count}.sh"""

                        os.system(cmd)
                        count += 1

          

    launch_jobs_with_node_limit(max_nodes=12)



# outer_optimizer.kwargs.quantization_bins={q} \
# outer_optimizer.kwargs.skip_norm_quantization=True \
# outer_optimizer.kwargs.simulate_quantization_after_reduce=True \
# outer_optimizer.kwargs.compressor_type=row_wise_statistical \

