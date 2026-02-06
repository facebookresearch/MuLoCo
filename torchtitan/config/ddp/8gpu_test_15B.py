optimizer = {'beta1': 0.9,
               'beta2': 0.99,
               'early_step_in_backward': False,
               'eps': 1e-08,
               'implementation': "for-loop",
               'lr': 0.0003,
               'name': 'AdamW',
               'weight_decay': 1e-4}



activation_checkpoint = {'mode': 'none', 'selective_ac_option': 'op'}

checkpoint = {'async_mode': 'async',
                'create_seed_checkpoint': False,
                'enable_checkpoint': False,
                'enable_first_step_checkpoint': False,
                'exclude_from_loading': [],
                'export_dtype': 'float32',
                'folder': 'checkpoint',
                'initial_load_model_weights_only': False,
                'delete_checkpoints_at_last_step': True,
                'initial_load_path': None,
                'interval': 10,
                'keep_latest_k': 3,
                'last_save_model_weights_only': False,
                'keep_selected_checkpoints': [], #[32, 64, 96, 128, 160, 192, 224, 256, 288, 384, 512, 640, 768, 896, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144],
                'load_step': -1}

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
                'warmup_steps': 10}

memory_estimation= {'disable_fake_mode': False, 'enabled': False},

metrics = {'disable_color_printing': False,
             'enable_tensorboard': True,
             'enable_wandb': True,
             'enable_lm_evaluation': False,
             'is_sweep': True,
             'log_all_ranks': False,
             'log_freq': 1,
             'save_for_all_ranks': False,
             'save_tb_folder': 'tb',
             'sweep_id': "mgyba376",
             'wandb_project': 'dil_nemotroncc_v2',
             'wandb_suffix': '_TEST-ONLY_MFU',
             'eval_freq': 30}

model = {'converters': [],
        #    'flavor': "llama3-w6144-d60-h48_qk_torch-rmsnorm_pa_pf",
        #    'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf',
        #    'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf',
          #  'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf',
        #    'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf',
        
          # 'flavor': "llama3-w3584-d42-h28_qk_torch-rmsnorm_pa_pf", #7B
        
          # 'flavor': "llama3-w4096-d48-h32_qk_torch-rmsnorm_pa_pf", #10B
          'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf',  #15B
        #   'flavor': "llama3-w5120-d60-h40_qk_torch-rmsnorm_pa_pf", #20B
        #   'flavor': 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf', #2B
           'name': 'llama3',
           'print_after_conversion': False,
           'tokenizer_path': './assets/tokenizer/Meta-Llama-3.1-8B/original/tokenizer.model'}

mx = {'filter_fqns': ['output'],
        'recipe_name': 'mxfp8',
        'use_fp8_dim1_cast_triton_kernel': True}


parallelism = {'context_parallel_degree': 1,
                 'context_parallel_rotate_method': 'allgather',
                 'data_parallel_replicate_degree': 1,
                 'data_parallel_shard_degree': 8,
                 'disable_loss_parallel': False,
                 'enable_async_tensor_parallel': False,
                 'enable_compiled_autograd': False,
                 'fsdp_reshard_after_forward': 'always',
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
              'global_batch_size': 64,
              'local_batch_size': 2,
              'eval_global_batch_size': 64,
              'eval_local_batch_size': 2,
              'train_num_workers': 4,
              'val_num_workers': 2,
              'max_norm': 1.0,
              'mixed_precision_param': 'bfloat16',
              'mixed_precision_reduce': 'float32',

              'seed': 42,
              'seq_len': 2048,
              'steps': 100}



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
