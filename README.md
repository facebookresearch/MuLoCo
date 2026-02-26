<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./media/muloco_logo_dark.svg">
    <img width="55%" src="./media/muloco_logo.svg" alt="MuLoCo">
  </picture>
</p>

<h3 align="center">
Muon is a Practical Inner Optimizer for DiLoCo
</h3>

<p align="center">
  | <a href="https://arxiv.org/abs/2505.23725"><b>Paper</b></a>
  | <a href="https://bentherien.github.io/muloco-1/"><b>Project Page</b></a>
  | <a href="https://github.com/bentherien/muloco-1"><b>MuLoCo-1 Code</b></a>
  | <a href="https://github.com/facebookresearch/MuLoCo"><b>Full Research Code</b></a>
  | <a href="https://x.com/twitter/status/2026862326184304734"><b>Tweet Thread</b></a>
  |
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.23725"><img src="https://img.shields.io/badge/arXiv-2505.23725-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/bentherien/MuLoCo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg" alt="PyTorch 2.7+"></a>
</p>

---

## Description

This repository contains the research code used to run all experiments presented in the paper [*MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo*](https://arxiv.org/abs/2505.23725). Our code bundles the following popular packages:

- **[torchtitan](https://github.com/pytorch/torchtitan)** - PyTorch native platform for training generative AI models
- **[torchft](https://github.com/meta-pytorch/torchft)** - Fault-tolerant training utilities
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** - Language model evaluation framework

Muon and AdamW implementations are in TorchTitan, while the outer optimizer and communication code is spread between the outer optimizer classes themselves and TorchFT. The code allows for WandB logging and evaluation during training via LM Eval Harness.

**NOTE:** We use a proprietary dataset internally at Meta which is not publicly available. As such, we have reverted the code to the default TorchTitan dataloader, but note that this is not what we used in our experiments.

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
export MULOCO_PATH=/path/to/MuLoCo
```

### 4. Create and sync the virtual environment

```bash
cd $MULOCO_PATH
uv sync
```

This will:
- Create a `.venv` virtual environment
- Install all dependencies from `pyproject.toml`
- Install local packages (torchtitan, torchft, lm-eval) in editable mode

<details>
<summary>Editable installations</summary>

The following local packages are installed in editable mode via `[tool.uv.sources]` in `pyproject.toml`:

| Package | Path | Description |
|---------|------|-------------|
| torchtitan | `torchtitan/` | PyTorch training framework |
| torchft | `torchft/` | Fault-tolerant training |
| lm-eval | `lm-evaluation-harness/` | Language model evaluation |

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

## Usage

### Activate the environment

```bash
export MULOCO_PATH=/path/to/MuLoCo
source $MULOCO_PATH/setup.sh
```

This will:
- Activate the UV virtual environment
- Configure PYTHONPATH for all repositories
- Set up WandB and HuggingFace credentials
- Configure Rust/Cargo if available

### Manual activation (alternative)

If you only need the virtual environment without additional setup:

```bash
source $MULOCO_PATH/.venv/bin/activate
```

## Directory Structure

```
MuLoCo/
├── pyproject.toml          # UV project configuration with all dependencies
├── setup.sh                # Environment setup script
├── install_dependencies.sh # Rust and protoc installer (no sudo)
├── README.md               # This file
├── torchtitan/             # PyTorch training framework
├── torchft/                # Fault-tolerant training
└── lm-evaluation-harness/  # LM evaluation
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
export MULOCO_PATH=/path/to/MuLoCo
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

## Citation

```bibtex
@article{therien2025muloco,
  title={MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo},
  author={Therien, Benjamin and Huang, Xiaolong and Defazio, Aaron and Rish, Irina and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2505.23725},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
