#!/bin/bash

# IdeaWeaver Environment Setup Script
# Creates a single virtual environment: ideaweaver-env with Python 3.12
# Handles Python version detection, installation, and dependency management

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions - redirect to stderr
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if git repository exists and clone if needed
check_and_clone_repo() {
    local repo_url="https://github.com/ideaweaver-ai-code/ideaweaver.git"
    local repo_dir="ideaweaver"
    
    # Check if we're already in the ideaweaver directory
    if [[ $(basename "$PWD") == "ideaweaver" ]]; then
        if [[ -d ".git" ]]; then
            log_info "Already in ideaweaver repository"
            return 0
        fi
    fi
    
    # Check if ideaweaver directory exists
    if [[ -d "$repo_dir" ]]; then
        log_warning "Directory '$repo_dir' already exists"
        if ask_confirmation "Remove existing directory and clone fresh?"; then
            rm -rf "$repo_dir"
        else
            log_info "Using existing directory"
            return 0
        fi
    fi
    
    # Check if git is installed
    if ! command_exists git; then
        error_exit "Git is not installed. Please install git first."
    fi
    
    # Clone the repository
    log_info "Cloning ideaweaver repository..."
    git clone "$repo_url" "$repo_dir" || error_exit "Failed to clone repository"
    
    # Change to the repository directory
    cd "$repo_dir" || error_exit "Failed to change to repository directory"
    
    log_success "Repository cloned successfully"
}

# Ask for user confirmation
ask_confirmation() {
    local message="$1"
    while true; do
        read -p "$(echo -e "${YELLOW}$message [y/N]:${NC} ")" yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            "" ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# Check Python version - must be exactly 3.12
check_python_version() {
    local python_cmd="$1"
    local required_version="$2"
    
    if command_exists "$python_cmd"; then
        local version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
        local major_minor=$(echo "$version" | cut -d'.' -f1,2)
        
        if [[ "$major_minor" == "$required_version" ]]; then
            return 0
        fi
    fi
    return 1
}

# Find Python 3.12
find_python312() {
    log_info "Searching for Python 3.12..."
    
    # Common Python 3.12 command variations
    local python_variants=("python3.12" "python3" "python")
    
    for cmd in "${python_variants[@]}"; do
        if check_python_version "$cmd" "3.12"; then
            local python_path=$(which $cmd)
            local version=$($cmd --version 2>&1 | cut -d' ' -f2)
            log_success "Found Python 3.12: $python_path (version $version)"
            echo "$python_path"
            return 0
        fi
    done
    
    return 1
}

# Install Python 3.12 on macOS
install_python312_macos() {
    log_info "Installing Python 3.12 on macOS..."
    
    if command_exists brew; then
        log_info "Using Homebrew to install Python 3.12..."
        brew install python@3.12 || error_exit "Failed to install Python 3.12 via Homebrew"
        
        # Add to PATH if needed
        local brew_python="/opt/homebrew/bin/python3.12"
        if [[ ! -x "$brew_python" ]]; then
            brew_python="/usr/local/bin/python3.12"
        fi
        
        if [[ -x "$brew_python" ]]; then
            log_success "Python 3.12 installed successfully: $brew_python"
            echo "$brew_python"
        else
            error_exit "Python 3.12 installation completed but binary not found"
        fi
    else
        log_error "Homebrew not found. Please install Homebrew first:"
        log_error "Visit: https://brew.sh/"
        error_exit "Cannot install Python 3.12 without Homebrew"
    fi
}

# Install Python 3.12 on Linux
install_python312_linux() {
    log_info "Installing Python 3.12 on Linux..."
    
    if command_exists apt-get; then
        log_info "Using apt to install Python 3.12..."
        sudo apt-get update || error_exit "Failed to update package list"
        sudo apt-get install -y python3.12 python3.12-venv python3.12-pip || error_exit "Failed to install Python 3.12"
        echo "python3.12"
    elif command_exists yum; then
        log_info "Using yum to install Python 3.12..."
        sudo yum install -y python3.12 || error_exit "Failed to install Python 3.12"
        echo "python3.12"
    elif command_exists dnf; then
        log_info "Using dnf to install Python 3.12..."
        sudo dnf install -y python3.12 || error_exit "Failed to install Python 3.12"
        echo "python3.12"
    else
        error_exit "No supported package manager found. Please install Python 3.12 manually."
    fi
}

# Get or install Python 3.12
get_python312() {
    local python312_cmd
    
    if python312_cmd=$(find_python312); then
        echo "$python312_cmd"
        return 0
    fi
    
    log_warning "Python 3.12 not found on system."
    
    if ask_confirmation "Would you like to install Python 3.12 automatically?"; then
        local os_type=$(detect_os)
        
        case "$os_type" in
            "macos")
                python312_cmd=$(install_python312_macos)
                ;;
            "linux")
                python312_cmd=$(install_python312_linux)
                ;;
            *)
                error_exit "Unsupported operating system: $os_type"
                ;;
        esac
        
        echo "$python312_cmd"
    else
        log_error "Python 3.12 is required for IdeaWeaver environment."
        log_error "Please install Python 3.12 manually and run this script again."
        exit 1
    fi
}

# Create virtual environment safely
create_venv() {
    local env_name="$1"
    local python_cmd="$2"
    
    log_info "Creating virtual environment: $env_name"
    
    # Remove existing environment if it exists
    if [[ -d "$env_name" ]]; then
        log_warning "Environment '$env_name' already exists."
        if ask_confirmation "Remove existing environment and recreate?"; then
            rm -rf "$env_name"
            log_info "Removed existing environment: $env_name"
        else
            log_info "Skipping environment creation for: $env_name"
            return 0
        fi
    fi
    
    # Create new environment using the system Python
    if [[ "$python_cmd" == *"ideaweaver-env"* ]]; then
        # If the Python path is inside the virtual environment, use the system Python
        python_cmd="python3.12"
    fi
    
    "$python_cmd" -m venv "$env_name" || error_exit "Failed to create virtual environment: $env_name"
    log_success "Created virtual environment: $env_name"
    
    # Verify environment
    if [[ ! -f "$env_name/bin/activate" ]]; then
        error_exit "Virtual environment created but activation script not found: $env_name/bin/activate"
    fi
}

# Install torch before requirements.txt if not already installed
install_torch() {
    if python -c "import torch" &>/dev/null; then
        log_info "PyTorch already installed, skipping."
        return
    fi

    if [[ -n "${CI:-}" ]]; then
        log_info "CI environment detected, installing CPU-only torch."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    elif command -v nvidia-smi &>/dev/null; then
        log_info "NVIDIA GPU detected. Please install the correct torch version for your CUDA version."
        log_info "See: https://pytorch.org/get-started/locally/"
        pip install torch  # This will install the default version, but user may want to override
    else
        log_info "No GPU detected, installing CPU-only torch."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
}

# Install packages in virtual environment
install_packages() {
    local env_name="$1"
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    log_info "Installing packages in environment: $env_name"
    
    # Activate environment
    source "$env_name/bin/activate" || error_exit "Failed to activate environment: $env_name"
    
    # Upgrade pip first
    pip install --upgrade pip || error_exit "Failed to upgrade pip in $env_name"
    
    # Install torch first
    install_torch

    # Install numpy before requirements to avoid build errors with auto-gptq
    pip install numpy

    # Force auto-gptq to build CPU-only (skip CUDA extensions)
    export FORCE_CUDA=0

    # Install from requirements.txt using --no-build-isolation
    local requirements_file="$script_dir/requirements.txt"
    if [[ -f "$requirements_file" ]]; then
        log_info "Installing from requirements.txt with --no-build-isolation"
        pip install --no-build-isolation -r "$requirements_file" || error_exit "Failed to install from requirements.txt"
    else
        error_exit "requirements.txt not found at: $requirements_file"
    fi

    # Create ideaweaver entry point
    log_info "Creating ideaweaver command entry point"
    local entry_point="$env_name/bin/ideaweaver"
    cat > "$entry_point" << 'EOF'
#!/bin/bash
source "$(dirname "$0")/activate"
python -c "from ideaweaver.cli import cli; cli()" "$@"
EOF
    chmod +x "$entry_point"
    
    # Deactivate environment
    deactivate
    
    log_success "Package installation completed for: $env_name"
}

# Main function
main() {
    log_info "🚀 Starting IdeaWeaver Environment Setup"
    log_info "This script will:"
    log_info "  • Clone the ideaweaver repository (if needed)"
    log_info "  • Create a virtual environment with Python 3.12"
    log_info "  • Install all required dependencies"
    
    if ! ask_confirmation "Continue with environment setup?"; then
        log_info "Setup cancelled by user"
        exit 0
    fi
    
    # Check and clone repository if needed
    check_and_clone_repo
    
    # Get Python 3.12
    local python312_cmd
    python312_cmd=$(get_python312) || error_exit "Failed to get Python 3.12"
    
    # Create virtual environment
    create_venv "ideaweaver-env" "$python312_cmd"
    
    # Install packages
    install_packages "ideaweaver-env"
    
    log_success "✅ IdeaWeaver environment setup completed successfully!"
    log_info "To activate the environment, run: source ideaweaver-env/bin/activate"
    log_info "To use ideaweaver commands, make sure you're in the ideaweaver directory"
}

# Run main function
main 