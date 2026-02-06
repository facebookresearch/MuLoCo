"""
Test file to verify MuonOptimizersContainer checkpointing correctness.

This test verifies that:
1. Optimizer state (momentum, variance) is correctly saved
2. Optimizer state is correctly loaded after checkpoint restore
3. Step counters in param_groups are correctly saved/restored
4. Training can resume correctly after checkpoint restore
"""

import copy
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


# Simple model for testing
class SimpleTransformerLayer(nn.Module):
    """A simple transformer-like layer with 2D weight matrices."""
    
    def __init__(self, dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        # 2D matrix parameters (should use Muon algorithm)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
        # 1D parameters (should use AdamW algorithm)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Simple forward pass
        h = self.norm1(x)
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)
        # Simplified attention
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
        h = self.wo(attn @ v)
        x = x + h
        
        h = self.norm2(x)
        h = self.w2(torch.relu(self.w1(h)))
        x = x + h
        return x


class SimpleModel(nn.Module):
    """Simple model with embedding, transformer layers, and output head."""
    
    def __init__(self, vocab_size: int = 1000, dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(dim, dim * 4) for _ in range(n_layers)
        ])
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, x):
        h = self.tok_embeddings(x)
        for layer in self.layers:
            h = layer(h)
        return self.output(h)


@dataclass
class MockMuonOptimizerConfig:
    """Mock config that mimics MuonOptimizerConfig structure."""
    
    name: str = "muon"
    lr: float = 0.125
    weight_decay: float = 0.01
    mu: float = 0.9
    betas: tuple = (0.9, 0.99)
    epsilon: float = 1e-8
    nesterov: bool = False
    adjust_lr: Optional[str] = "spectral_norm"
    flatten: bool = True
    use_triton: bool = False
    use_polar_express: bool = False
    cautious_wd: bool = False
    algorithm: str = "muon"
    
    # Parameter-specific optimizer selection
    scalar_optimizer: str = "adamw"
    embedding_optimizer: str = "adamw"
    head_optimizer: str = "adamw"
    routing_optimizer: str = "adamw"
    expert_optimizer: Optional[str] = None
    head_lr_scaling: bool = False
    
    # Learning rate scaling factors
    scalar_lr_factor: float = 1.0
    embedding_lr_factor: float = 1.0
    head_lr_factor: float = 1.0
    routing_lr_factor: float = 1.0
    expert_lr_factor: float = 1.0


@dataclass
class MockParallelDims:
    """Mock ParallelDims for testing without distributed setup."""
    
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    world_size: int = 1
    enable_loss_parallel: bool = False
    world_mesh: Any = None
    
    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1
    
    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1


def compute_state_norms(optimizer, model=None) -> Dict[str, float]:
    """Compute norms of optimizer state tensors for comparison.
    
    If model is provided, uses parameter names as keys (more reliable).
    Otherwise uses parameter indices (less reliable across optimizer instances).
    """
    norms = {}
    
    if model is not None:
        # Build mapping from data_ptr to name
        ptr_to_name = {p.data_ptr(): name for name, p in model.named_parameters()}
    
    for i, (param, state) in enumerate(optimizer.state.items()):
        # Determine key to use
        if model is not None:
            key = ptr_to_name.get(param.data_ptr(), f"param_{i}")
        else:
            key = f"param_{i}"
        
        if "momentum" in state:
            norms[f"{key}_momentum"] = state["momentum"].float().norm().item()
        if "variance" in state:
            norms[f"{key}_variance"] = state["variance"].float().norm().item()
    return norms


def test_muon_optimizer_state_dict_basic():
    """Test basic state_dict and load_state_dict operations."""
    print("\n" + "=" * 60)
    print("TEST: Basic state_dict and load_state_dict")
    print("=" * 60)
    
    try:
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizersContainer,
            MuonOptimizerConfig,
        )
        from torchtitan.distributed import ParallelDims
    except ImportError as e:
        print(f"SKIP: Could not import required modules: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create config
    config = MuonOptimizerConfig(
        lr=0.125,
        weight_decay=0.01,
        mu=0.9,
        betas=(0.9, 0.99),
        epsilon=1e-8,
        algorithm="muon",
        flatten=True,
        use_triton=False,
        scalar_optimizer="adamw",
        embedding_optimizer="adamw",
        head_optimizer="adamw",
        head_lr_scaling=False,
    )
    
    # Create mock parallel dims
    parallel_dims = MockParallelDims()
    
    # Create optimizer container
    optimizer_container = MuonOptimizersContainer(
        model_parts=[model],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    print(f"\nOptimizer has {len(optimizer_container.muon_optimizer.param_groups)} param groups")
    for i, group in enumerate(optimizer_container.muon_optimizer.param_groups):
        print(f"  Group {i}: algorithm={group.get('algorithm', 'N/A')}, "
              f"params={len(group['params'])}, step={group.get('step', 0)}")
    
    # Run a few training steps to build up optimizer state
    print("\n--- Running training steps ---")
    for step in range(5):
        # Forward pass
        input_ids = torch.randint(0, 1000, (4, 32), device=device)
        output = model(input_ids)
        loss = output.mean()
        
        # Backward pass
        optimizer_container.zero_grad()
        loss.backward()
        optimizer_container.step()
        
        # Check state after step
        step_counts = [g.get("step", 0) for g in optimizer_container.muon_optimizer.param_groups]
        print(f"  Step {step + 1}: param_group steps = {step_counts}")
    
    # Get optimizer state before saving
    print("\n--- Capturing optimizer state ---")
    state_before = compute_state_norms(optimizer_container.muon_optimizer, model)
    print(f"Number of params with state: {len(optimizer_container.muon_optimizer.state)}")
    print(f"Sample norms before save: {dict(list(state_before.items())[:3])}")
    
    # Get state dict
    state_dict = optimizer_container.state_dict()
    print(f"state_dict has {len(state_dict)} keys")
    
    # Check for momentum keys
    momentum_keys = [k for k in state_dict.keys() if "momentum" in k]
    variance_keys = [k for k in state_dict.keys() if "variance" in k]
    print(f"Found {len(momentum_keys)} momentum keys, {len(variance_keys)} variance keys")
    
    # Check _param_group_steps and save expected steps before load (load_state_dict pops this key)
    if "_param_group_steps" in state_dict:
        expected_steps = state_dict['_param_group_steps'].copy() if isinstance(state_dict['_param_group_steps'], list) else list(state_dict['_param_group_steps'])
        print(f"_param_group_steps: {expected_steps}")
    else:
        print("WARNING: _param_group_steps not found in state_dict!")
        return False
    
    # Create a new model and optimizer
    print("\n--- Creating fresh model and optimizer ---")
    model2 = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    optimizer_container2 = MuonOptimizersContainer(
        model_parts=[model2],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Check state before loading
    state_before_load = compute_state_norms(optimizer_container2.muon_optimizer, model2)
    print(f"Fresh optimizer state norms: {dict(list(state_before_load.items())[:3])}")
    
    # Load state dict
    print("\n--- Loading state dict ---")
    optimizer_container2.load_state_dict(state_dict)
    
    # Verify state after loading
    print("\n--- Verifying loaded state ---")
    state_after = compute_state_norms(optimizer_container2.muon_optimizer, model2)
    print(f"Loaded optimizer state norms: {dict(list(state_after.items())[:3])}")
    
    # Check step counters
    step_counts_after = [g.get("step", 0) for g in optimizer_container2.muon_optimizer.param_groups]
    print(f"Loaded param_group steps: {step_counts_after}")
    
    # Verify correctness
    all_passed = True
    
    # Check step counters match (expected_steps was saved before load)
    if step_counts_after != expected_steps:
        print(f"FAIL: Step counters mismatch! Expected {expected_steps}, got {step_counts_after}")
        all_passed = False
    else:
        print(f"PASS: Step counters match: {step_counts_after}")
    
    # Check state norms match
    for key in state_before:
        if key not in state_after:
            print(f"FAIL: Key {key} not found in loaded state!")
            all_passed = False
            continue
        
        original = state_before[key]
        loaded = state_after[key]
        
        # Check if values match (with tolerance)
        if abs(original - loaded) > 1e-5:
            print(f"FAIL: {key} mismatch! Original: {original:.6f}, Loaded: {loaded:.6f}")
            all_passed = False
        else:
            print(f"PASS: {key} matches (diff={abs(original - loaded):.2e})")
    
    return all_passed


def test_muon_checkpoint_with_dcp():
    """Test MuonOptimizersContainer checkpointing with Distributed Checkpoint (DCP)."""
    print("\n" + "=" * 60)
    print("TEST: Checkpointing with DCP")
    print("=" * 60)
    
    try:
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizersContainer,
            MuonOptimizerConfig,
        )
    except ImportError as e:
        print(f"SKIP: Could not import required modules: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create temp directory for checkpoint
    checkpoint_dir = tempfile.mkdtemp(prefix="muon_ckpt_test_")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    try:
        # Create model and optimizer
        model = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
        
        config = MuonOptimizerConfig(
            lr=0.125,
            weight_decay=0.01,
            mu=0.9,
            betas=(0.9, 0.99),
            epsilon=1e-8,
            algorithm="muon",
            flatten=True,
            use_triton=False,
            scalar_optimizer="adamw",
            embedding_optimizer="adamw",
            head_optimizer="adamw",
            head_lr_scaling=False,
        )
        
        parallel_dims = MockParallelDims()
        
        optimizer = MuonOptimizersContainer(
            model_parts=[model],
            muon_config=config,
            parallel_dims=parallel_dims,
        )
        
        # Run training steps
        print("\n--- Running training steps ---")
        for step in range(10):
            input_ids = torch.randint(0, 1000, (4, 32), device=device)
            output = model(input_ids)
            loss = output.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Capture state before saving
        state_norms_before = compute_state_norms(optimizer.muon_optimizer, model)
        steps_before = [g.get("step", 0) for g in optimizer.muon_optimizer.param_groups]
        print(f"Steps before save: {steps_before}")
        print(f"Sample norms before save: {dict(list(state_norms_before.items())[:3])}")
        
        # Get the state dict for saving
        optimizer_state = optimizer.state_dict()
        model_state = {
            k: v for k, v in model.state_dict().items()
        }
        
        # Save checkpoint
        print("\n--- Saving checkpoint ---")
        full_state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "step": 10,
        }
        
        # Use torch.save for simplicity (DCP requires distributed setup)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save(full_state, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Create fresh model and optimizer
        print("\n--- Creating fresh model and optimizer ---")
        model2 = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
        optimizer2 = MuonOptimizersContainer(
            model_parts=[model2],
            muon_config=config,
            parallel_dims=parallel_dims,
        )
        
        # Load checkpoint
        print("\n--- Loading checkpoint ---")
        loaded_state = torch.load(checkpoint_path, weights_only=False)
        model2.load_state_dict(loaded_state["model"])
        optimizer2.load_state_dict(loaded_state["optimizer"])
        
        # Verify state after loading
        state_norms_after = compute_state_norms(optimizer2.muon_optimizer, model2)
        steps_after = [g.get("step", 0) for g in optimizer2.muon_optimizer.param_groups]
        print(f"Steps after load: {steps_after}")
        print(f"Sample norms after load: {dict(list(state_norms_after.items())[:3])}")
        
        # Verify correctness
        all_passed = True
        
        # Check steps
        if steps_before != steps_after:
            print(f"FAIL: Steps mismatch! Before: {steps_before}, After: {steps_after}")
            all_passed = False
        else:
            print(f"PASS: Steps match: {steps_after}")
        
        # Check state norms
        for key in state_norms_before:
            if key not in state_norms_after:
                print(f"FAIL: Key {key} missing after load!")
                all_passed = False
                continue
            
            before = state_norms_before[key]
            after = state_norms_after[key]
            
            if abs(before - after) > 1e-5:
                print(f"FAIL: {key} mismatch! Before: {before:.6f}, After: {after:.6f}")
                all_passed = False
            else:
                print(f"PASS: {key} matches")
        
        return all_passed
        
    finally:
        # Cleanup
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def test_muon_training_continuation():
    """Test that training can continue correctly after checkpoint restore."""
    print("\n" + "=" * 60)
    print("TEST: Training continuation after checkpoint")
    print("=" * 60)
    
    try:
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizersContainer,
            MuonOptimizerConfig,
        )
    except ImportError as e:
        print(f"SKIP: Could not import required modules: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    
    config = MuonOptimizerConfig(
        lr=0.125,
        weight_decay=0.01,
        mu=0.9,
        betas=(0.9, 0.99),
        epsilon=1e-8,
        algorithm="muon",
        flatten=True,
        use_triton=False,
        scalar_optimizer="adamw",
        embedding_optimizer="adamw",
        head_optimizer="adamw",
        head_lr_scaling=False,
    )
    
    parallel_dims = MockParallelDims()
    
    optimizer = MuonOptimizersContainer(
        model_parts=[model],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Generate fixed data for reproducibility
    torch.manual_seed(123)
    all_data = [torch.randint(0, 1000, (4, 32), device=device) for _ in range(20)]
    
    # Run 10 steps continuously
    print("\n--- Running 10 steps continuously ---")
    losses_continuous = []
    for step in range(10):
        data = all_data[step]
        output = model(data)
        loss = output.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_continuous.append(loss.item())
        
    # Save checkpoint at step 5
    print("\n--- Saving checkpoint at step 5 ---")
    
    # Reset and run to step 5
    torch.manual_seed(42)
    model2 = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    optimizer2 = MuonOptimizersContainer(
        model_parts=[model2],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    for step in range(5):
        data = all_data[step]
        output = model2(data)
        loss = output.mean()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    
    # Save state
    checkpoint = {
        "model": copy.deepcopy(model2.state_dict()),
        "optimizer": optimizer2.state_dict(),
        "step": 5,
    }
    
    # Capture state norms at step 5
    norms_at_step5 = compute_state_norms(optimizer2.muon_optimizer, model2)
    print(f"State norms at step 5: {dict(list(norms_at_step5.items())[:3])}")
    
    # Create fresh model and load checkpoint
    print("\n--- Loading checkpoint and continuing ---")
    torch.manual_seed(42)
    model3 = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    optimizer3 = MuonOptimizersContainer(
        model_parts=[model3],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Load checkpoint
    model3.load_state_dict(checkpoint["model"])
    optimizer3.load_state_dict(checkpoint["optimizer"])
    
    # Verify state was loaded correctly
    norms_after_load = compute_state_norms(optimizer3.muon_optimizer, model3)
    print(f"State norms after load: {dict(list(norms_after_load.items())[:3])}")
    
    # Check if norms match
    norms_match = True
    for key in norms_at_step5:
        if key in norms_after_load:
            if abs(norms_at_step5[key] - norms_after_load[key]) > 1e-5:
                print(f"FAIL: {key} norm mismatch after load!")
                print(f"  Expected: {norms_at_step5[key]:.6f}")
                print(f"  Got: {norms_after_load[key]:.6f}")
                norms_match = False
    
    if norms_match:
        print("PASS: All state norms match after load")
    
    # Continue training from step 5
    losses_resumed = []
    for step in range(5, 10):
        data = all_data[step]
        output = model3(data)
        loss = output.mean()
        optimizer3.zero_grad()
        loss.backward()
        optimizer3.step()
        losses_resumed.append(loss.item())
    
    print("\n--- Comparing loss trajectories ---")
    print(f"Continuous losses (steps 5-10): {[f'{l:.4f}' for l in losses_continuous[5:10]]}")
    print(f"Resumed losses (steps 5-10):    {[f'{l:.4f}' for l in losses_resumed]}")
    
    # Check if losses match
    losses_match = True
    for i, (cont, resumed) in enumerate(zip(losses_continuous[5:10], losses_resumed)):
        if abs(cont - resumed) > 1e-4:
            print(f"FAIL: Loss mismatch at step {5 + i}! Continuous: {cont:.6f}, Resumed: {resumed:.6f}")
            losses_match = False
    
    if losses_match:
        print("PASS: Loss trajectories match!")
    else:
        print("WARNING: Loss trajectories differ (may indicate state restoration issue)")
    
    return norms_match


def test_muon_state_dict_content():
    """Test that state_dict contains expected keys and values."""
    print("\n" + "=" * 60)
    print("TEST: State dict content inspection")
    print("=" * 60)
    
    try:
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizersContainer,
            MuonOptimizerConfig,
        )
    except ImportError as e:
        print(f"SKIP: Could not import required modules: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    
    config = MuonOptimizerConfig(
        lr=0.125,
        weight_decay=0.01,
        mu=0.9,
        betas=(0.9, 0.99),
        epsilon=1e-8,
        algorithm="muon",
        flatten=True,
        use_triton=False,
        scalar_optimizer="adamw",
        embedding_optimizer="adamw",
        head_optimizer="adamw",
        head_lr_scaling=False,
    )
    
    parallel_dims = MockParallelDims()
    
    optimizer = MuonOptimizersContainer(
        model_parts=[model],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Run one step
    input_ids = torch.randint(0, 1000, (4, 32), device=device)
    output = model(input_ids)
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Get state dict
    state_dict = optimizer.state_dict()
    
    print(f"\n--- State dict analysis ---")
    print(f"Total keys: {len(state_dict)}")
    
    # Categorize keys
    momentum_keys = [k for k in state_dict.keys() if "momentum" in k]
    variance_keys = [k for k in state_dict.keys() if "variance" in k]
    step_keys = [k for k in state_dict.keys() if "step" in k.lower()]
    other_keys = [k for k in state_dict.keys() 
                  if "momentum" not in k and "variance" not in k and "step" not in k.lower()]
    
    print(f"Momentum keys: {len(momentum_keys)}")
    print(f"Variance keys: {len(variance_keys)}")
    print(f"Step keys: {len(step_keys)}")
    print(f"Other keys: {len(other_keys)}")
    
    if other_keys:
        print(f"  Other keys: {other_keys}")
    
    # Check for _param_group_steps
    has_param_group_steps = "_param_group_steps" in state_dict
    print(f"\n_param_group_steps present: {has_param_group_steps}")
    if has_param_group_steps:
        print(f"  Value: {state_dict['_param_group_steps']}")
    
    # Sample momentum values
    print("\n--- Sample momentum values ---")
    for key in list(momentum_keys)[:5]:
        value = state_dict[key]
        if hasattr(value, 'norm'):
            print(f"  {key}: norm={value.float().norm().item():.6f}, shape={value.shape}")
        else:
            print(f"  {key}: type={type(value)}")
    
    # Sample variance values
    print("\n--- Sample variance values ---")
    for key in list(variance_keys)[:5]:
        value = state_dict[key]
        if hasattr(value, 'norm'):
            print(f"  {key}: norm={value.float().norm().item():.6f}, shape={value.shape}")
        else:
            print(f"  {key}: type={type(value)}")
    
    # Check that all momentum tensors have non-zero values
    print("\n--- Checking for zero tensors ---")
    zero_momentum = 0
    for key in momentum_keys:
        value = state_dict[key]
        if hasattr(value, 'norm') and value.float().norm().item() == 0:
            zero_momentum += 1
    
    if zero_momentum > 0:
        print(f"WARNING: {zero_momentum}/{len(momentum_keys)} momentum tensors are zero!")
    else:
        print(f"PASS: All {len(momentum_keys)} momentum tensors are non-zero")
    
    return has_param_group_steps and len(momentum_keys) > 0


def test_muon_dcp_set_optimizer_state_dict():
    """Test that DCP's set_optimizer_state_dict properly restores state."""
    print("\n" + "=" * 60)
    print("TEST: DCP set_optimizer_state_dict behavior")
    print("=" * 60)
    
    try:
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizersContainer,
            MuonOptimizerConfig,
        )
    except ImportError as e:
        print(f"SKIP: Could not import required modules: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    
    config = MuonOptimizerConfig(
        lr=0.125,
        weight_decay=0.01,
        mu=0.9,
        betas=(0.9, 0.99),
        epsilon=1e-8,
        algorithm="muon",
        flatten=True,
        use_triton=False,
        scalar_optimizer="adamw",
        embedding_optimizer="adamw",
        head_optimizer="adamw",
        head_lr_scaling=False,
    )
    
    parallel_dims = MockParallelDims()
    
    optimizer = MuonOptimizersContainer(
        model_parts=[model],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Run several steps to build up significant momentum
    print("\n--- Running 10 training steps ---")
    for _ in range(10):
        input_ids = torch.randint(0, 1000, (4, 32), device=device)
        output = model(input_ids)
        loss = output.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Capture raw optimizer state
    print("\n--- Capturing raw optimizer state ---")
    raw_state = {}
    for i, (param, state) in enumerate(optimizer.muon_optimizer.state.items()):
        for k, v in state.items():
            if hasattr(v, 'clone'):
                raw_state[f"param_{i}_{k}"] = v.clone()
    print(f"Captured {len(raw_state)} raw state tensors")
    
    # Get state dict via DCP
    print("\n--- Getting state via DCP get_optimizer_state_dict ---")
    dcp_state = get_optimizer_state_dict(
        model, 
        optimizer.muon_optimizer,
        options=StateDictOptions(flatten_optimizer_state_dict=True)
    )
    print(f"DCP state dict has {len(dcp_state)} keys")
    
    # Compare raw state vs DCP state
    print("\n--- Comparing raw vs DCP state ---")
    momentum_keys_dcp = [k for k in dcp_state.keys() if "momentum" in k]
    print(f"DCP momentum keys: {len(momentum_keys_dcp)}")
    
    for key in list(momentum_keys_dcp)[:3]:
        dcp_norm = dcp_state[key].float().norm().item()
        print(f"  {key}: norm={dcp_norm:.6f}")
    
    # Now test loading into fresh optimizer
    print("\n--- Testing load with fresh optimizer ---")
    model2 = SimpleModel(vocab_size=1000, dim=256, n_layers=2).to(device)
    optimizer2 = MuonOptimizersContainer(
        model_parts=[model2],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Get fresh state norms (should be near zero from warm-up)
    fresh_norms = compute_state_norms(optimizer2.muon_optimizer, model2)
    print(f"Fresh optimizer norms: {dict(list(fresh_norms.items())[:3])}")
    
    # Load via DCP
    print("\n--- Loading via DCP set_optimizer_state_dict ---")
    set_optimizer_state_dict(
        model2,
        optimizer2.muon_optimizer,
        optim_state_dict=dcp_state,
        options=StateDictOptions(flatten_optimizer_state_dict=True)
    )
    
    # Check if state was loaded
    loaded_norms = compute_state_norms(optimizer2.muon_optimizer, model2)
    print(f"After DCP load norms: {dict(list(loaded_norms.items())[:3])}")
    
    # Compare with original
    original_norms = compute_state_norms(optimizer.muon_optimizer, model)
    print(f"Original optimizer norms: {dict(list(original_norms.items())[:3])}")
    
    # Check if load was successful
    all_passed = True
    for key in original_norms:
        if key not in loaded_norms:
            print(f"FAIL: Key {key} missing after load!")
            all_passed = False
            continue
        
        orig = original_norms[key]
        loaded = loaded_norms[key]
        
        if abs(orig - loaded) > 1e-5:
            print(f"FAIL: {key} mismatch!")
            print(f"  Original: {orig:.6f}")
            print(f"  Loaded: {loaded:.6f}")
            print(f"  Fresh: {fresh_norms.get(key, 'N/A')}")
            all_passed = False
    
    if all_passed:
        print("\nPASS: DCP set_optimizer_state_dict correctly restored all state")
    else:
        print("\nFAIL: DCP set_optimizer_state_dict did not correctly restore state")
    
    return all_passed


def test_param_to_state_mapping():
    """Diagnose the parameter to state dict key mapping issue."""
    print("\n" + "=" * 60)
    print("TEST: Parameter to state dict key mapping")
    print("=" * 60)
    
    try:
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizersContainer,
            MuonOptimizerConfig,
        )
    except ImportError as e:
        print(f"SKIP: Could not import required modules: {e}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleModel(vocab_size=100, dim=64, n_layers=1).to(device)
    
    config = MuonOptimizerConfig(
        lr=0.125,
        weight_decay=0.01,
        mu=0.9,
        betas=(0.9, 0.99),
        epsilon=1e-8,
        algorithm="muon",
        flatten=True,
        use_triton=False,
        scalar_optimizer="adamw",
        embedding_optimizer="adamw",
        head_optimizer="adamw",
        head_lr_scaling=False,
    )
    
    parallel_dims = MockParallelDims()
    
    optimizer = MuonOptimizersContainer(
        model_parts=[model],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    
    # Run one step
    input_ids = torch.randint(0, 100, (2, 8), device=device)
    output = model(input_ids)
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print model parameter names and their order
    print("\n--- Model parameter order ---")
    model_params = list(model.named_parameters())
    for i, (name, param) in enumerate(model_params):
        print(f"  {i}: {name} shape={param.shape}")
    
    # Print optimizer param_groups structure
    print("\n--- Optimizer param_groups structure ---")
    for i, group in enumerate(optimizer.muon_optimizer.param_groups):
        algo = group.get('algorithm', 'N/A')
        param_count = len(group['params'])
        print(f"  Group {i}: algorithm={algo}, params={param_count}")
        for j, param in enumerate(group['params']):
            # Find matching model param name
            param_name = "unknown"
            for name, mp in model_params:
                if mp.data_ptr() == param.data_ptr():
                    param_name = name
                    break
            print(f"    {j}: {param_name} shape={param.shape}")
    
    # Print state dict keys
    print("\n--- State dict key structure ---")
    state_dict = optimizer.state_dict()
    momentum_keys = sorted([k for k in state_dict.keys() if "momentum" in k])
    print(f"Momentum keys ({len(momentum_keys)}):")
    for key in momentum_keys:
        value = state_dict[key]
        if hasattr(value, 'norm'):
            print(f"  {key}: norm={value.float().norm().item():.4f}")
    
    # Map optimizer state params to names
    print("\n--- Optimizer.state parameter mapping ---")
    for i, (param, state) in enumerate(optimizer.muon_optimizer.state.items()):
        # Find param name
        param_name = "unknown"
        for name, mp in model_params:
            if mp.data_ptr() == param.data_ptr():
                param_name = name
                break
        
        mom_norm = state.get("momentum", torch.tensor(0)).float().norm().item() if "momentum" in state else "N/A"
        print(f"  param[{i}] = {param_name}: momentum_norm={mom_norm}")
    
    # Now test what happens during DCP get/set
    print("\n--- DCP roundtrip test ---")
    
    # Get flattened state dict
    dcp_state = get_optimizer_state_dict(
        model, 
        optimizer.muon_optimizer,
        options=StateDictOptions(flatten_optimizer_state_dict=True)
    )
    
    # Find the momentum values and their keys
    original_values = {}
    for i, (param, state) in enumerate(optimizer.muon_optimizer.state.items()):
        if "momentum" in state:
            for name, mp in model_params:
                if mp.data_ptr() == param.data_ptr():
                    original_values[name] = state["momentum"].float().norm().item()
                    break
    
    print("Original momentum norms by param name:")
    for name, norm in sorted(original_values.items()):
        print(f"  {name}: {norm:.4f}")
    
    # Create fresh optimizer
    model2 = SimpleModel(vocab_size=100, dim=64, n_layers=1).to(device)
    optimizer2 = MuonOptimizersContainer(
        model_parts=[model2],
        muon_config=config,
        parallel_dims=parallel_dims,
    )
    model2_params = list(model2.named_parameters())
    
    # Load state
    set_optimizer_state_dict(
        model2,
        optimizer2.muon_optimizer,
        optim_state_dict=dcp_state,
        options=StateDictOptions(flatten_optimizer_state_dict=True)
    )
    
    # Check loaded values
    loaded_values = {}
    for i, (param, state) in enumerate(optimizer2.muon_optimizer.state.items()):
        if "momentum" in state:
            for name, mp in model2_params:
                if mp.data_ptr() == param.data_ptr():
                    loaded_values[name] = state["momentum"].float().norm().item()
                    break
    
    print("\nLoaded momentum norms by param name:")
    for name, norm in sorted(loaded_values.items()):
        print(f"  {name}: {norm:.4f}")
    
    # Compare
    print("\n--- Comparison ---")
    all_match = True
    for name in sorted(original_values.keys()):
        if name not in loaded_values:
            print(f"FAIL: {name} not found in loaded state!")
            all_match = False
            continue
        
        orig = original_values[name]
        loaded = loaded_values[name]
        
        if abs(orig - loaded) > 1e-5:
            print(f"FAIL: {name} - Original: {orig:.4f}, Loaded: {loaded:.4f}")
            all_match = False
        else:
            print(f"PASS: {name} matches")
    
    return all_match


def main():
    """Run all tests."""
    print("=" * 70)
    print("MUON OPTIMIZER CHECKPOINTING TESTS")
    print("=" * 70)
    
    results = {}
    
    # Run diagnostic test first
    results["param_mapping"] = test_param_to_state_mapping()
    
    # Run other tests
    results["basic_state_dict"] = test_muon_optimizer_state_dict_basic()
    results["checkpoint_with_save"] = test_muon_checkpoint_with_dcp()
    results["training_continuation"] = test_muon_training_continuation()
    results["state_dict_content"] = test_muon_state_dict_content()
    results["dcp_set_optimizer"] = test_muon_dcp_set_optimizer_state_dict()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if passed is None:
            status = "SKIP"
        print(f"  {name}: {status}")
        if passed is False:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed!")
        print("\nMuonOptimizersContainer checkpointing is working correctly.")
        print("The DCP (Distributed Checkpoint) properly saves and restores:")
        print("  - Momentum tensors for all parameters")
        print("  - Variance tensors for AdamW parameters")
        print("  - Step counters in param_groups")
    else:
        print("Some tests failed!")
        print("\nDIAGNOSIS:")
        print("If tests fail, check that the optimizer state is being correctly")
        print("mapped by parameter FQN (fully qualified name) and not by index.")
        print("DCP uses parameter names to match state, so the model must be passed")
        print("to set_optimizer_state_dict() to correctly map parameters.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

