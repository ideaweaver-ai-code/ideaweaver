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
    
    # Create new environment
    "$python_cmd" -m venv "$env_name" || error_exit "Failed to create virtual environment: $env_name"
    log_success "Created virtual environment: $env_name"
    
    # Verify environment
    if [[ ! -f "$env_name/bin/activate" ]]; then
        error_exit "Virtual environment created but activation script not found: $env_name/bin/activate"
    fi
}

# Install packages in virtual environment
install_packages() {
    local env_name="$1"
    local requirements_file="$2"
    
    log_info "Installing packages in environment: $env_name"
    
    # Activate environment
    source "$env_name/bin/activate" || error_exit "Failed to activate environment: $env_name"
    
    # Upgrade pip first
    pip install --upgrade pip || error_exit "Failed to upgrade pip in $env_name"
    
    # Install from requirements file if provided
    if [[ -n "$requirements_file" && -f "$requirements_file" ]]; then
        log_info "Installing from requirements file: $requirements_file"
        pip install -r "$requirements_file" || error_exit "Failed to install from $requirements_file"
    fi
    
    # Install IdeaWeaver in development mode if backend exists
    if [[ -d "backend" && -f "backend/setup.py" ]]; then
        log_info "Installing IdeaWeaver in development mode..."
        cd backend
        pip install -e . || log_warning "Failed to install IdeaWeaver in development mode"
        cd ..
    fi
    
    # Deactivate environment
    deactivate
    
    log_success "Package installation completed for: $env_name"
}

# Create requirements file
create_requirements_file() {
    log_info "Using consolidated requirements.txt file..."
    
    # Verify requirements file exists
    if [[ ! -f "requirements.txt" ]]; then
        error_exit "requirements.txt not found. Please ensure you're in the correct directory."
    fi
    
    log_success "Requirements file verified"
}

# Setup IdeaWeaver environment
setup_ideaweaver_env() {
    log_info "Setting up IdeaWeaver environment with Python 3.12..."
    
    local python312_cmd=$(get_python312)
    create_venv "ideaweaver-env" "$python312_cmd"
    install_packages "ideaweaver-env" "requirements.txt"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test IdeaWeaver environment
    log_info "Testing IdeaWeaver environment..."
    source "ideaweaver-env/bin/activate"
    if python -c "import click, yaml, requests, rich, torch, transformers; print('✅ IdeaWeaver dependencies OK')" 2>/dev/null; then
        log_success "IdeaWeaver environment verification passed"
    else
        log_warning "Some dependencies may not be installed correctly"
    fi
    
    # Check if ideaweaver CLI is available
    if command_exists ideaweaver; then
        log_success "IdeaWeaver CLI is available"
        ideaweaver --version 2>/dev/null || log_info "IdeaWeaver CLI installed but version check failed"
    else
        log_warning "IdeaWeaver CLI not found - this may be normal if setup.py installation failed"
    fi
    
    deactivate
}

# Main execution
main() {
    log_info "🚀 Starting IdeaWeaver Environment Setup"
    log_info "This script will create a virtual environment with Python 3.12:"
    log_info "  • ideaweaver-env (Python 3.12 with all dependencies)"
    echo
    
    if ! ask_confirmation "Continue with environment setup?"; then
        log_info "Setup cancelled by user."
        exit 0
    fi
    
    # Create requirements file
    create_requirements_file
    
    # Setup environment
    setup_ideaweaver_env
    
    # Verify installation
    verify_installation
    
    # Final instructions
    echo
    log_success "🎉 Environment setup completed successfully!"
    echo
    log_info "📋 Next steps:"
    log_info "  1. Activate environment: source ideaweaver-env/bin/activate"
    log_info "  2. Set API keys as needed (OpenAI, HuggingFace, etc.):"
    log_info "     export OPENAI_API_KEY='your-openai-key'"
    log_info "     export HUGGINGFACE_HUB_TOKEN='your-hf-token'"
    log_info "  3. Test installation: ideaweaver --help"
    echo
    log_info "📁 Created files:"
    log_info "  • ideaweaver-env/ (virtual environment)"
    echo
    log_success "🚀 IdeaWeaver is ready to use!"
}

# Trap errors and cleanup
trap 'log_error "Script interrupted or failed. Check the error messages above."' ERR

# Run main function
main "$@" 