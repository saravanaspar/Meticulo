#!/bin/bash
#
# Meticulo Installer
# Usage: curl -fsSL https://meticulo.ai/install.sh | bash
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if grep -q "microsoft" /proc/version 2>/dev/null; then
            echo "wsl"
        else
            echo "linux"
        fi
    else
        echo "unsupported"
    fi
}

# Check Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)')
        if [ "$PYTHON_VERSION" -ge 310 ]; then
            PYTHON_CMD="python3"
            return 0
        fi
    fi
    
    # Try to install Python on Linux
    if [ "$OS" = "linux" ] || [ "$OS" = "wsl" ]; then
        log_info "Installing Python..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm python python-pip
        fi
        
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
            return 0
        fi
    fi
    
    log_error "Python 3.10+ not found. Please install Python from https://python.org"
    exit 1
}

# Check required tools
check_tools() {
    local missing=()
    
    for tool in git cmake g++; do
        if ! command -v $tool &> /dev/null; then
            missing+=($tool)
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_warn "Missing tools: ${missing[*]}"
        log_info "Installing build dependencies..."
        
        if [ "$OS" = "linux" ] || [ "$OS" = "wsl" ]; then
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y git cmake g++ make ninja-build
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y git cmake gcc-g++ make
            fi
        elif [ "$OS" = "macos" ]; then
            if command -v brew &> /dev/null; then
                brew install git cmake make
            else
                log_error "Please install Homebrew from https://brew.sh"
                exit 1
            fi
        fi
    fi
}

# Install Meticulo
install_meticulo() {
    log_info "Installing Meticulo..."
    
    # Create venv
    if [ -d "$HOME/.meticulo-venv" ]; then
        log_info "Using existing virtual environment"
    else
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$HOME/.meticulo-venv"
    fi
    
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    # Activate and install
    source "$HOME/.meticulo-venv/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    # Install from local source
    pip install -e "$SCRIPT_DIR"
    
    # Create symlink
    if [ -L "$HOME/.local/bin/meticulo" ]; then
        rm "$HOME/.local/bin/meticulo"
    elif [ -f "$HOME/.local/bin/meticulo" ]; then
        log_warn "Found existing meticulo in ~/.local/bin, backing up..."
        mv "$HOME/.local/bin/meticulo" "$HOME/.local/bin/meticulo.bak"
    fi
    
    mkdir -p "$HOME/.local/bin"
    ln -s "$HOME/.meticulo-venv/bin/meticulo" "$HOME/.local/bin/meticulo"
    
    # Add to PATH if not already there
    if [[ ":$PATH:" != *"$HOME/.local/bin"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    log_info "Meticulo installed successfully!"
}

# Build engine
build_engine() {
    log_info "Building Meticulo engine (this may take 10-20 minutes)..."
    
    source "$HOME/.meticulo-venv/bin/activate"
    meticulo setup
    
    if [ $? -eq 0 ]; then
        log_info "Engine built successfully!"
    else
        log_error "Engine build failed. Try: meticulo setup --force"
    fi
}

# Optional: Build TUI
build_tui() {
    if ! command -v cargo &> /dev/null; then
        log_warn "Cargo not found. Skipping TUI build."
        log_info "To build TUI later: cargo build --release --manifest-path tui/Cargo.toml"
        return
    fi
    
    log_info "Building TUI (this may take 5-10 minutes)..."
    
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR"
    
    if [ -d "tui" ]; then
        cargo build --release --manifest-path tui/Cargo.toml
        log_info "TUI built at: ./tui/target/release/meticulo-tui"
    fi
}

# Main
main() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║       METICULO INSTALLER             ║"
    echo "╚══════════════════════════════════════╝"
    echo ""
    
    OS=$(detect_os)
    if [ "$OS" = "unsupported" ]; then
        log_error "Unsupported OS. Meticulo supports Linux, macOS, and WSL."
        exit 1
    fi
    
    log_info "Detected: $OS"
    
    check_python
    check_tools
    install_meticulo
    
    # Ask about engine and TUI
    if [ "$1" = "--with-tui" ]; then
        build_engine
        build_tui
    elif [ "$1" = "--minimal" ]; then
        log_info "Minimal install complete. Run 'meticulo setup' later to build engine."
    else
        log_info "Building engine..."
        build_engine
    fi
    
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  Installation complete!             ║"
    echo "╚══════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  meticulo info           # Check system"
    echo "  meticulo pull <model>   # Download a model"
    echo "  meticulo serve --model <model>  # Start server"
    echo "  meticulo tui            # Launch TUI"
    echo ""
}

main "$@"