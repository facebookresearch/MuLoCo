# MuLoCo

This directory contains a UV-based installation of our research code for training MuLoCo 

- **torchtitan** - PyTorch native platform for training generative AI models
- **torchft** - Fault-tolerant training utilities
- **lm-evaluation-harness** - Language model evaluation framework

## Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- Rust (for building torchft)
- Protocol Buffers compiler (`protoc`) - required for torchft

## Installation

### 1. Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Rust and Protocol Buffers (required for torchft)

Run the provided installation script (no sudo required):

```bash
cd $MULOCO_PATH
./install_dependencies.sh
```

This script will:
- Install Rust via rustup (if not already installed)
- Download and install `protoc` from GitHub releases to `~/.local/`
- Automatically detect your OS and architecture (Linux/macOS, x86_64/arm64)

After running, add the following to your `~/.bashrc` or `~/.zshrc`:
```bash
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env"
```

<details>
<summary>Manual installation (alternative)</summary>

**Rust:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Protocol Buffers (`protoc`):**
```bash
mkdir -p $HOME/.local/bin
PROTOC_VERSION=29.3
curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip
unzip protoc-${PROTOC_VERSION}-linux-x86_64.zip -d $HOME/.local/protoc
ln -sf $HOME/.local/protoc/bin/protoc $HOME/.local/bin/protoc
export PATH="$HOME/.local/bin:$PATH"
rm protoc-${PROTOC_VERSION}-linux-x86_64.zip
```

</details>

### 3. Set required environment variables

```bash
export MULOCO_PATH=/home/btherien/full_muloco_install
export DATA_PATH=/path/to/your/nemotroncc_mixed  # Path to NemotronCC dataset
```

### 4. Create and sync the virtual environment

```bash
cd $MULOCO_PATH
uv sync
```

This will:
- Create a `.venv` virtual environment
- Install all dependencies from `pyproject.toml`
- Install local packages (torchtitan, torchft, amaia, xlformers, lm-eval) in editable mode

<details>
<summary>Editable installations</summary>

The following local packages are installed in editable mode via `[tool.uv.sources]` in `pyproject.toml`:

| Package | Path | Description |
|---------|------|-------------|
| torchtitan | `torchtitan/` | PyTorch training framework |
| torchft | `torchft/` | Fault-tolerant training |
| lm-eval | `lm-evaluation-harness/` | Language model evaluation |
| amaia | `amaia/` | AMAIA codebase |
| xlformers | `xlformers/` | XLFormers utilities |

To manually install lm-evaluation-harness in editable mode (standalone):
```bash
cd $MULOCO_PATH
uv pip install -e lm-evaluation-harness
```

To install with optional dependencies (e.g., math tasks):
```bash
uv pip install -e "lm-evaluation-harness[math]"
```

</details>

### 5. Install TorchFT and lm eval harness


```bash
cd $MULOCO_PATH
uv pip install -e torchft[dev]
uv pip install -e lm-evaluation-harness
```

### 6. Set up dataset symbolic links

The training code expects the NemotronCC dataset to be available via symbolic links. Use the provided helper script:

```bash
# Make sure both environment variables are set
# export MULOCO_PATH=
# export DATA_PATH=

# Run the dataset setup script
$MULOCO_PATH/setup_dataset.sh
```

This creates:
- **Training data**: 19 symlinks in `torchtitan/data/nemotron_cc_mixed/train/` (splits 5%-100%)
- **Validation data**: 1 symlink in `torchtitan/data/nemotron_cc_mixed/val/` (split 0%-5%)

<details>
<summary>Manual setup (alternative)</summary>

If you prefer to set up symlinks manually:

```bash
# Create dataset directories
mkdir -p $MULOCO_PATH/torchtitan/data/nemotron_cc_mixed/train
mkdir -p $MULOCO_PATH/torchtitan/data/nemotron_cc_mixed/val

# Create training data symlinks
for pct in 005.0_010.0 010.0_015.0 015.0_020.0 020.0_025.0 025.0_030.0 \
           030.0_035.0 035.0_040.0 040.0_045.0 045.0_050.0 050.0_055.0 \
           055.0_060.0 060.0_065.0 065.0_070.0 070.0_075.0 075.0_080.0 \
           080.0_085.0 085.0_090.0 090.0_095.0 095.0_100.0; do
    ln -sf $DATA_PATH/split_${pct}.jsonl $MULOCO_PATH/torchtitan/data/nemotron_cc_mixed/train/
done

# Create validation data symlink
ln -sf $DATA_PATH/split_000.0_005.0.jsonl $MULOCO_PATH/torchtitan/data/nemotron_cc_mixed/val/
```

</details>

### 7. Install CUDA-specific dependencies (if needed)

Some CUDA-specific packages may need manual installation:

```bash
# xformers (if using internal version)
uv pip install git+ssh://git@github.com/fairinternal/xformers.git@910de3ab888ae5ab5c9b4c482fd7d4f1e03886c3

# fgcuda (internal package - install from wheel if available)
# uv pip install /path/to/fgcuda-0.0.1-cp311-cp311-linux_x86_64.whl
```

## Usage

### Activate the environment

```bash
export MULOCO_PATH=/home/btherien/full_muloco_install
source $MULOCO_PATH/setup.sh
```

This will:
- Activate the UV virtual environment
- Configure PYTHONPATH for all repositories
- Set up WandB and HuggingFace credentials
- Configure Rust/Cargo if available
- Warn if dataset symlinks are not set up

### Manual activation (alternative)

If you only need the virtual environment without additional setup:

```bash
source /home/btherien/full_muloco_install/.venv/bin/activate
```

## Directory Structure

```
full_muloco_install/
├── pyproject.toml          # UV project configuration with all dependencies
├── setup.sh                # Environment setup script
├── setup_dataset.sh        # Dataset symlink setup script
├── install_dependencies.sh # Rust and protoc installer (no sudo)
├── README.md               # This file
├── torchtitan/         # PyTorch training framework
│   └── data/
│       └── nemotron_cc_mixed/
│           ├── train/  # Training data symlinks (19 files)
│           └── val/    # Validation data symlink (1 file)
├── torchft/            # Fault-tolerant training
├── lm-evaluation-harness/  # LM evaluation
├── amaia/              # AMAIA codebase
└── xlformers/          # XLFormers utilities
```

## Optional Dependencies

Install optional dependency groups as needed:

```bash
# Development tools
uv sync --extra dev

# Math evaluation tasks
uv sync --extra math

# IFEval tasks
uv sync --extra ifeval

# vLLM support
uv sync --extra vllm

# Multilingual support
uv sync --extra multilingual
```

## Troubleshooting

### MULOCO_PATH not set
If you see "ERROR: MULOCO_PATH environment variable is not set", run:
```bash
export MULOCO_PATH=/home/btherien/full_muloco_install
```

### DATA_PATH not set
If you need to set up the dataset symlinks, ensure DATA_PATH points to your NemotronCC data:
```bash
export DATA_PATH=/path/to/your/nemotroncc_mixed
$MULOCO_PATH/setup_dataset.sh
```

### Virtual environment not found
If the `.venv` directory doesn't exist, create it with:
```bash
cd $MULOCO_PATH && uv sync
```

### Rust or protoc not found (for torchft)
If you see errors about missing `rustc`, `cargo`, or `protoc`, run the installation script:
```bash
cd $MULOCO_PATH
./install_dependencies.sh
source $HOME/.cargo/env
export PATH="$HOME/.local/bin:$PATH"
```

Alternatively, set the `PROTOC` environment variable directly if protoc is installed elsewhere:
```bash
export PROTOC=/path/to/protoc
```

### Dataset symlinks broken
If the training data symlinks are broken, verify that `DATA_PATH` points to the correct location and re-run:
```bash
export DATA_PATH=/path/to/your/nemotroncc_mixed
$MULOCO_PATH/setup_dataset.sh
```
