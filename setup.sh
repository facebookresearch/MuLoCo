#!/bin/bash
# MULOCO Environment Setup Script
# This script sets up the environment for the MULOCO training framework
# using UV for package management

# ============================================
# Check for required MULOCO_PATH environment variable
# ============================================
if [ -z "$MULOCO_PATH" ]; then
    echo -e "\033[1;31m"
    echo "████████████████████████████████████████████████████████████████"
    echo "██                                                            ██"
    echo "██  ERROR: MULOCO_PATH environment variable is not set.      ██"
    echo "██                                                            ██"
    echo "██  Please set MULOCO_PATH to the installation directory:    ██"
    echo "██    export MULOCO_PATH=/home/btherien/full_muloco_install  ██"
    echo "██                                                            ██"
    echo "████████████████████████████████████████████████████████████████"
    echo -e "\033[0m"
    return 1 2>/dev/null || exit 1
fi

# Verify the path exists
if [ ! -d "$MULOCO_PATH" ]; then
    echo "ERROR: MULOCO_PATH directory does not exist: $MULOCO_PATH"
    return 1 2>/dev/null || exit 1
fi

echo "Setting up MULOCO environment from: $MULOCO_PATH"

# ============================================
# Activate UV virtual environment
# ============================================
if [ -d "$MULOCO_PATH/.venv" ]; then
    source "$MULOCO_PATH/.venv/bin/activate"
    echo "Activated UV virtual environment"
else
    echo "WARNING: Virtual environment not found at $MULOCO_PATH/.venv"
    echo "Run 'cd $MULOCO_PATH && uv sync' to create it first."
fi

# ============================================
# Check for dataset symlinks
# ============================================
DATASET_DIR="$MULOCO_PATH/torchtitan/data/nemotron_cc_mixed"
if [ ! -d "$DATASET_DIR/train" ] || [ ! -d "$DATASET_DIR/val" ]; then
    echo -e "\033[1;33m"
    echo "WARNING: Dataset directories not found at $DATASET_DIR"
    echo ""
    echo "To set up the dataset, run:"
    echo "  export DATA_PATH=/path/to/your/nemotroncc_mixed"
    echo "  # Then follow the symlink setup instructions in README.md"
    echo -e "\033[0m"
fi

# ============================================
# WandB Configuration
# ============================================
export WANDB_API_KEY=
export WANDB_DIR=/checkpoint/optim/btherien/wandb

# ============================================
# PYTHONPATH Configuration
# ============================================
# Clear existing PYTHONPATH to avoid conflicts
export PYTHONPATH=""

# Add xlformers and its submodules
export PYTHONPATH="$MULOCO_PATH/xlformers:$PYTHONPATH"
export PYTHONPATH="$MULOCO_PATH/xlformers/core:$PYTHONPATH"
export PYTHONPATH="$MULOCO_PATH/xlformers/core/tokenizers:$PYTHONPATH"

# Add amaia
export PYTHONPATH="$MULOCO_PATH/amaia:$PYTHONPATH"

# Add torchtitan (if needed for direct imports)
export PYTHONPATH="$MULOCO_PATH/torchtitan:$PYTHONPATH"

# Add torchft (if needed for direct imports)
export PYTHONPATH="$MULOCO_PATH/torchft:$PYTHONPATH"

# Add lm-evaluation-harness (if needed for direct imports)
export PYTHONPATH="$MULOCO_PATH/lm-evaluation-harness:$PYTHONPATH"

# ============================================
# HuggingFace Configuration
# ============================================
export HF_TOKEN=
export HF_DATASETS_OFFLINE=1

# ============================================
# Rust/Cargo Configuration (for torchft)
# ============================================
# Rust is required for building torchft
# To configure your current shell, source the cargo env:
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi

# ============================================
# Optional Debug Settings (uncomment as needed)
# ============================================
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export NCCL_DEBUG=INFO
echo -e "\033[1;32m"
echo "███╗   ███╗██╗   ██╗██╗      ██████╗  ██████╗ ██████╗ "
echo "████╗ ████║██║   ██║██║     ██╔═══██╗██╔════╝██╔═══██╗"
echo "██╔████╔██║██║   ██║██║     ██║   ██║██║     ██║   ██║"
echo "██║╚██╔╝██║██║   ██║██║     ██║   ██║██║     ██║   ██║"
echo "██║ ╚═╝ ██║╚██████╔╝███████╗╚██████╔╝╚██████╗╚██████╔╝"
echo "╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ "
echo ""
echo "🚀 MULOCO ENVIRONMENT SETUP COMPLETE! 🚀"
echo ""
echo "📁 MULOCO_PATH: $MULOCO_PATH"
echo "🐍 PYTHONPATH configured for:"
echo "   • xlformers"
echo "   • amaia" 
echo "   • torchtitan"
echo "   • torchft"
echo "   • lm-evaluation-harness"
echo ""
echo "✅ Ready for training!"
echo -e "\033[0m"
