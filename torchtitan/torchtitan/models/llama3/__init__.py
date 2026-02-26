# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_llama
from .infra.pipeline import pipeline_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


llama3_configs = {
    #0.5X multiplier 
    "llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf_no-flex": TransformerModelArgs(
        dim=512, n_layers=6, n_heads=4, rope_theta=500000,
        use_flex_attn=False, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    "llama3-w512-d6-h4_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=512, n_layers=6, n_heads=4, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 1X multiplier
    "llama3-w1024-d12-h8_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=1024, n_layers=12, n_heads=8, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    #1.5X multiplier 
    "llama3-w1536-d18-h12_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=1536, n_layers=18, n_heads=12, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    #2X multiplier 
    "llama3-w2048-d24-h16_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=2048, n_layers=24, n_heads=16, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),
    # 2.5X multiplier
    "llama3-w2560-d30-h20_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=2560, n_layers=30, n_heads=20, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 3.0X multiplier
    "llama3-w3072-d36-h24_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=3072, n_layers=36, n_heads=24, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 3.5X multiplier
    "llama3-w3584-d42-h28_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=3584, n_layers=42, n_heads=28, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 4.0X multiplier 10B 
    "llama3-w4096-d48-h32_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=4096, n_layers=48, n_heads=32, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 4.5X multiplier 15B 
    "llama3-w4608-d54-h36_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=4608, n_layers=54, n_heads=36, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 5.0X multiplier 20B 
    "llama3-w5120-d60-h40_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=5120, n_layers=60, n_heads=40, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 5.5X multiplier 27B
    "llama3-w5632-d66-h44_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=5632, n_layers=66, n_heads=44, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

    # 6.0X multiplier 34B
    "llama3-w6144-d72-h48_qk_torch-rmsnorm_pa_pf": TransformerModelArgs(
        dim=6144, n_layers=72, n_heads=48, rope_theta=500000,
        use_flex_attn=True, attn_mask_type="block_causal",
        qk_norm=True, qk_impl="torch_rmsnorm",
        ffn_dim_multiplier=2.75,
        use_post_attn_norm=True,
        use_post_ffn_norm=True,
        norm_eps=1e-6,
    ),

}

from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
register_train_spec(
    TrainSpec(
        name="llama3",
        cls=Transformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)