#!/bin/bash
# Install dependencies required for building torchft (Rust and protoc)
# This script does not require sudo - all installations are local to $HOME

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PROTOC_VERSION="${PROTOC_VERSION:-29.3}"
LOCAL_BIN="$HOME/.local/bin"
LOCAL_DIR="$HOME/.local"

# Detect OS and architecture
detect_platform() {
    local os=""
    local arch=""
    
    case "$(uname -s)" in
        Linux*)  os="linux";;
        Darwin*) os="osx";;
        *)       error "Unsupported OS: $(uname -s)"; exit 1;;
    esac
    
    case "$(uname -m)" in
        x86_64)  arch="x86_64";;
        aarch64) arch="aarch_64";;
        arm64)   arch="aarch_64";;
        *)       error "Unsupported architecture: $(uname -m)"; exit 1;;
    esac
    
    echo "${os}-${arch}"
}

# Install Rust using rustup
install_rust() {
    if command -v rustc &> /dev/null && command -v cargo &> /dev/null; then
        info "Rust is already installed: $(rustc --version)"
        return 0
    fi
    
    info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Source cargo environment
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    if command -v rustc &> /dev/null; then
        info "Rust installed successfully: $(rustc --version)"
    else
        error "Failed to install Rust"
        exit 1
    fi
}

# Install protoc from GitHub releases
install_protoc() {
    # Check if protoc is already installed and working
    if command -v protoc &> /dev/null; then
        info "protoc is already installed: $(protoc --version)"
        return 0
    fi
    
    info "Installing Protocol Buffers compiler (protoc) version ${PROTOC_VERSION}..."
    
    local platform=$(detect_platform)
    local zip_file="protoc-${PROTOC_VERSION}-${platform}.zip"
    local download_url="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${zip_file}"
    local install_dir="${LOCAL_DIR}/protoc"
    
    # Create directories
    mkdir -p "$LOCAL_BIN"
    mkdir -p "$install_dir"
    
    # Create temp directory for download
    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT
    
    info "Downloading from: $download_url"
    if ! curl -fsSL -o "$tmp_dir/$zip_file" "$download_url"; then
        error "Failed to download protoc. Check if version ${PROTOC_VERSION} exists for platform ${platform}"
        error "Available releases: https://github.com/protocolbuffers/protobuf/releases"
        exit 1
    fi
    
    info "Extracting to: $install_dir"
    if ! unzip -q -o "$tmp_dir/$zip_file" -d "$install_dir"; then
        error "Failed to extract protoc"
        exit 1
    fi
    
    # Create symlink in local bin
    ln -sf "$install_dir/bin/protoc" "$LOCAL_BIN/protoc"
    
    # Verify installation
    if [ -x "$LOCAL_BIN/protoc" ]; then
        info "protoc installed successfully: $($LOCAL_BIN/protoc --version)"
    else
        error "Failed to install protoc"
        exit 1
    fi
}

# Update PATH if needed
setup_path() {
    if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
        export PATH="$LOCAL_BIN:$PATH"
        warn "Added $LOCAL_BIN to PATH for this session"
        warn "Add the following to your ~/.bashrc or ~/.zshrc for persistence:"
        echo ""
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
    fi
    
    # Source cargo env if available
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
}

# Fix Python dependency version conflicts
# tyro 0.9.x requires typeguard>=4.0.0, but some packages install typeguard 2.x
fix_python_deps() {
    local muloco_path="${MULOCO_PATH:-$(dirname "$(readlink -f "$0")")}"
    local venv_path="$muloco_path/.venv"
    
    if [ ! -d "$venv_path" ]; then
        warn "Virtual environment not found at $venv_path"
        warn "Run 'uv sync' first, then re-run this script to fix Python dependencies"
        return 0
    fi
    
    info "Fixing Python dependency versions..."
    
    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        warn "uv not found, skipping Python dependency fix"
        return 0
    fi
    
    # Fix typeguard version (tyro requires typeguard>=4.0.0)
    local current_typeguard=$(cd "$muloco_path" && uv pip show typeguard 2>/dev/null | grep "^Version:" | awk '{print $2}')
    if [ -n "$current_typeguard" ]; then
        local major_version=$(echo "$current_typeguard" | cut -d. -f1)
        if [ "$major_version" -lt 4 ]; then
            info "Upgrading typeguard from $current_typeguard to >=4.0.0 (required by tyro)..."
            (cd "$muloco_path" && uv pip install "typeguard>=4.0.0")
            info "typeguard upgraded successfully"
        else
            info "typeguard version $current_typeguard is compatible"
        fi
    fi
}

# Main
main() {
    echo "=============================================="
    echo "  MULOCO Dependencies Installer"
    echo "=============================================="
    echo ""
    
    # Setup PATH first so we can find any existing installations
    setup_path
    
    # Install dependencies
    install_rust
    echo ""
    install_protoc
    echo ""
    
    # Fix Python dependency conflicts if venv exists
    fix_python_deps
    echo ""
    
    # Final verification
    echo "=============================================="
    echo "  Installation Summary"
    echo "=============================================="
    
    local all_ok=true
    
    if command -v rustc &> /dev/null; then
        info "✓ Rust: $(rustc --version)"
    else
        error "✗ Rust: NOT FOUND"
        all_ok=false
    fi
    
    if command -v cargo &> /dev/null; then
        info "✓ Cargo: $(cargo --version)"
    else
        error "✗ Cargo: NOT FOUND"
        all_ok=false
    fi
    
    if command -v protoc &> /dev/null; then
        info "✓ protoc: $(protoc --version)"
    else
        error "✗ protoc: NOT FOUND"
        all_ok=false
    fi
    
    echo ""
    
    if $all_ok; then
        info "All dependencies installed successfully!"
        echo ""
        info "Next steps:"
        echo "  1. Source your shell config or run: source \$HOME/.cargo/env"
        echo "  2. Ensure PATH includes \$HOME/.local/bin"
        echo "  3. Run: cd \$MULOCO_PATH && uv sync"
        echo "  4. Re-run this script to fix Python dependency versions"
        echo "     (or run: uv pip install 'typeguard>=4.0.0')"
    else
        error "Some dependencies failed to install. Please check the errors above."
        exit 1
    fi
}

main "$@"

