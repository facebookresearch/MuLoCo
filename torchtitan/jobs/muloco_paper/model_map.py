MODEL_STEP_MAP = {'llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf': {
  32: 46080,
  64: 23040,
  128: 11520,
  256: 5760,
  512: 2880,
  1024: 1440,
  2048: 720,
  4096: 360},
 'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf': {
  32: 130560,
  64: 65280,
  128: 32640,
  256: 16320,
  512: 8160,
  1024: 4080,
  2048: 2040,
  4096: 1020},
 'llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf': {
  64: 140160,
  128: 70080,
  256: 35040,
  512: 17520,
  1024: 8760,
  2048: 4380,
  4096: 2190},
 'llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf': {64: 268800,
  128: 134400,
  256: 67200,
  512: 33600,
  1024: 16800,
  2048: 8400,
  4096: 4200,
  8192: 2100},
 'llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf': {64: 468480,
  128: 234240,
  256: 117120,
  512: 58560,
  1024: 29280,
  2048: 14640,
  4096: 7320,
  8192: 3660},
 'llama3-w3072-d36-h24_qk_torch-rmsnorm_pa_pf': {64: 756480,
  128: 378240,
  256: 189120,
  512: 94560,
  1024: 47280,
  2048: 23640,
  4096: 11820},
 'llama3-w3584-d42-h28_qk_torch-rmsnorm_pa_pf': {64: 1150080,
  128: 575040,
  256: 287520,
  512: 143760,
  1024: 71880,
  2048: 35940,
  4096: 17970},
 'llama3-w4096-d48-h32_qk_torch-rmsnorm_pa_pf': {64: 1666560,
  128: 833280,
  256: 416640,
  512: 208320,
  1024: 104160,
  2048: 52080,
  4096: 26040},
  
 'llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf': {32: 4650240,
 64: 2325120,
 128: 1162560,
 256: 581280,
 512: 290640,
 1024: 145320,
 2048: 72660,
 4096: 36330,
 8192: 18165,
 16384: 9082,
 32768: 4541,
 65536: 2270}
  
}




LBS_MAP = {
    'llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf': 32,
    'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf': 16,
    'llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf': 16,
    'llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf': 8,
    'llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf': 4,
    'llama3-w3072-d36-h24_qk_torch-rmsnorm_pa_pf': 2,
    'llama3-w3584-d42-h28_qk_torch-rmsnorm_pa_pf': 2,
    'llama3-w4096-d48-h32_qk_torch-rmsnorm_pa_pf': 2,
    'llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf': 2,
}



USE_DP_PARALLELISM_MAP = {
    'llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf': True,
    'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf': True,
    'llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf': False,
    'llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf': False,
    'llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf': False,
    'llama3-w3072-d36-h24_qk_torch-rmsnorm_pa_pf': False,
    'llama3-w3584-d42-h28_qk_torch-rmsnorm_pa_pf': False,
    'llama3-w4096-d48-h32_qk_torch-rmsnorm_pa_pf': False,
    'llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf': False,
}



OPTIMAL_WD_MAP = {
    'llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf': {"muon": 0.01, "adamw": 0.001},
    'llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf': {"muon": 0.01, "adamw": 0.01},
    'llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf': {"muon": 0.01, "adamw": 0.01},
    'llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf': {"muon": 0.001, "adamw": 0.001},
    'llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf': {"muon": 0.001, "adamw": 0.001},
    'llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf': {"muon": 0.0001, "adamw": 0.0001},
}