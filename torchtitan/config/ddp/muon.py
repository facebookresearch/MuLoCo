optimizer = {
'beta1': 0.9,
'beta2': 0.99,
'early_step_in_backward': False,
'eps': 1e-8,
'implementation': 'fused',
'lr': 0.01,
'name': 'Muon',
'weight_decay': 0.0001,

# Muon-specific parameters
'mu': 0.9,                     # Momentum factor for Muon
'algorithm': 'muon',           # Main algorithm to use for 2D matrices
'nesterov': False,             # Whether to use Nesterov momentum
'adjust_lr': 'spectral_norm',  # How to adjust LR: "spectral_norm", "rms_norm", or None (commented for debugging)
'flatten': True,               # Whether to flatten 3D+ tensors to 2D (needed for MoE experts)
'use_triton': True,           # Whether to use Triton kernel for Newton-Schulz

# Parameter-specific optimizer selection
'scalar_optimizer': 'adamw',        # For 1D parameters (biases, layer norms)
'embedding_optimizer': 'adamw',     # For embedding layers
'head_optimizer': 'adamw',          # For model head/output layers
'routing_optimizer': 'adamw',       # For routing layers (DeepSeek MoE)
'head_lr_scaling': False,            # Apply 1/sqrt(dim) scaling to head layers
# 'expert_optimizer': 'adamw',        # Comment out to let experts use default classification (Muon for 2D matrices)

# Learning rate scaling factors
'scalar_lr_factor': 1.0,            # LR multiplier for scalar parameters
'embedding_lr_factor': 1.0,         # LR multiplier for embedding parameters
'head_lr_factor': 1.0,              # LR multiplier for head parameters (after head_lr_scaling)
'routing_lr_factor': 1.0,
               }


activation_checkpoint = {'mode': 'none', 'selective_ac_option': 'op'}

checkpoint = {'async_mode': 'async',
                'create_seed_checkpoint': False,
                'enable_checkpoint': False,
                'enable_first_step_checkpoint': False,
                'exclude_from_loading': [],
                'export_dtype': 'float32',
                'folder': 'checkpoint',
                'initial_load_model_weights_only': False,
                'initial_load_path': "outputs/geud794f_DP_Muon_LR0.011049_b10.9_b20.99_wd0.0004_steps72661_dps8_dpr128_cp1_tp1_pp1_B2048_SL2048_llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf/checkpoint/step-72661",
                'interval': 2000,
                'keep_latest_k': 3,
                'keep_selected_checkpoints': [20000,35000,55000,72660],
                'last_save_model_weights_only': False,
                'load_step': -1}


# checkpoint = {'async_mode': 'async',
#                 'create_seed_checkpoint': False,
#                 'enable_checkpoint': True,
#                 'enable_first_step_checkpoint': False,
#                 'exclude_from_loading': [],
#                 'export_dtype': 'float32',
#                 'folder': 'checkpoint',
#                 'initial_load_model_weights_only': True,
#                 'initial_load_path': None,
#                 'interval': 1000,
#                 'keep_latest_k': 3,
#                 'keep_selected_checkpoints': [120, 135, 150, 180, 360, 375, 390, 420, 720, 735, 750, 780, 1440, 1455, 1470, 1500, 3840, 3855, 3870, 3900, 4320, 4335, 4350, 4380, 5760, 5775, 5790, 5820],
#                 'last_save_model_weights_only': False,
#                 'load_step': -1}

comm = {'init_timeout_seconds': 300,
          'trace_buf_size': 20000,
          'train_timeout_seconds': 100}

experimental = {'custom_args_module': '', 'custom_import': ''}

fault_tolerance = {'enable': False,
                     'fragment_sync_delay': 0,
                     'fragment_update_alpha': 0.0,
                     'group_size': 0,
                     'min_replica_size': 1,
                     'replica_id': 0,
                     'semi_sync_method': None,
                     'should_quantize': False,
                     'sync_steps': 30}

float8 = {'emulate': False,
            'enable_fsdp_float8_all_gather': False,
            'filter_fqns': ['output'],
            'force_recompute_fp8_weight_in_bwd': False,
            'moe_fqns_prototype': [],
            'precompute_float8_dynamic_scale_for_fsdp': False,
            'recipe_name': None}

job = {'config_file': './torchtitan/models/llama3/train_configs/diloco_180m.toml',
         'description': 'diloco 180M baseline SL512 BS512',
         'dump_folder': './outputs',
         'print_args': False,
         'use_for_integration_test': False}

lr_scheduler= {'decay_ratio': None,
                'decay_type': 'cosine',
                'lr_min': 0.1,
                'warmup_steps': 300}

memory_estimation= {'disable_fake_mode': False, 'enabled': False},

metrics = {'disable_color_printing': False,
             'enable_tensorboard': True,
             'enable_wandb': True,
             'is_sweep': True,
             'log_all_ranks': False,
             'log_freq': 1,
             'save_for_all_ranks': False,
             'save_tb_folder': 'tb',
             'sweep_id': "mgyba376",
             'wandb_project': 'dil_nemotroncc_v2',
             'wandb_suffix': '',
             'eval_freq': 500}

model = {'converters': [],
           'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf',
           'name': 'llama3',
           'print_after_conversion': False,
           'tokenizer_path': './assets/tokenizer/Meta-Llama-3.1-8B/original/tokenizer.model'}

mx = {'filter_fqns': ['output'],
        'recipe_name': 'mxfp8',
        'use_fp8_dim1_cast_triton_kernel': True}


parallelism = {'context_parallel_degree': 1,
                 'context_parallel_rotate_method': 'allgather',
                 'data_parallel_replicate_degree': 8,
                 'data_parallel_shard_degree': 1,
                 'disable_loss_parallel': False,
                 'enable_async_tensor_parallel': False,
                 'enable_compiled_autograd': False,
                 'fsdp_reshard_after_forward': 'never',
                 'pipeline_parallel_degree': 1,
                 'pipeline_parallel_layers_per_stage': None,
                 'pipeline_parallel_microbatch_size': 1,
                 'pipeline_parallel_schedule': '1F1B',
                 'pipeline_parallel_schedule_csv': '',
                 'pipeline_parallel_split_points': [],
                 'tensor_parallel_degree': 1}

profiling = {'enable_memory_snapshot': False,
               'enable_profiling': False,
               'profile_freq': 100,
               'save_memory_snapshot_folder': 'memory_snapshot',
               'save_traces_folder': 'profile_trace'}

training = {'compile': True,
              'dataset': 'dclm',
              'dataset_path': None,
              'deterministic': False,
              'enable_cpu_offload': False,
              'gc_debug': False,
              'gc_freq': 50,
              'global_batch_size': 512,
              'local_batch_size': 16,
              'eval_global_batch_size': 512,
              'eval_local_batch_size': 16,
              'train_num_workers': 8,
              'val_num_workers': 2,
              'max_norm': 1.0,
              'mixed_precision_param': 'bfloat16',
              'mixed_precision_reduce': 'float32',

              'seed': 42,
              'seq_len': 2048,
              'steps': 10000}

data_train = {'sources': [{'path': 'data/nemotron_cc_mixed/train', 'weight': 1,
                        'pattern': '(?s:.*\\.jsonl)\\Z',}],
 'tokenizer': {'name': 'tiktoken',
               'path': 'assets/tokenizer/Meta-Llama-3.1-8B/original/tokenizer.model'}}


data_val = [{'name': 'nemotroncc',
  'sources': [{'path': 'data/nemotron_cc_mixed/val',
                         'pattern': '(?s:.*\\.jsonl)\\Z',
                         'weight': 1}],
 'tokenizer': {'name': 'tiktoken',
               'path': 'assets/tokenizer/Meta-Llama-3.1-8B/original/tokenizer.model'}}]