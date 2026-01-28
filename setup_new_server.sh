#!/bin/bash

# Setup script for neural audio codec project on new server
# This script creates a virtual environment and installs all required dependencies

set -e  # Exit on error

PROJECT_DIR="/mnt/Data/muaw1874/audio_cod"
VENV_NAME="venv"

echo "=========================================="
echo "Neural Audio Codec - New Server Setup"
echo "=========================================="
echo ""

# Check if we're in the project directory
if [ ! -f "conversation_context.md" ]; then
    echo "ERROR: conversation_context.md not found!"
    echo "Please run this script from the project directory: $PROJECT_DIR"
    exit 1
fi

echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check if Python 3.8+ is available
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Check if venv already exists
if [ -d "$VENV_NAME" ]; then
    echo "WARNING: Virtual environment '$VENV_NAME' already exists!"
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_NAME"
    else
        echo "Keeping existing virtual environment and updating packages..."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Step 2: Creating virtual environment..."
    python3 -m venv "$VENV_NAME"
    echo "✓ Virtual environment created"
    echo ""
else
    echo "Step 2: Using existing virtual environment"
    echo ""
fi

# Activate virtual environment
echo "Step 3: Activating virtual environment..."
source "$VENV_NAME/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Step 5: Installing project dependencies..."
echo "This may take several minutes (especially PyTorch with CUDA)..."
echo ""

if [ -f "current_requirements.txt" ]; then
    pip install -r current_requirements.txt
    echo "✓ All dependencies installed from current_requirements.txt"
else
    echo "WARNING: current_requirements.txt not found!"
    echo "Installing core dependencies manually..."
    
    # Install PyTorch with CUDA support first (largest dependency)
    pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu124
    
    # Install other critical dependencies
    pip install pesq==0.0.4 pystoi==0.4.1
    pip install soundfile==0.13.1 sounddevice==0.5.5
    pip install pyyaml==6.0.3 tqdm==4.67.1
    pip install matplotlib==3.10.8 scipy==1.15.3
    pip install tensorboard==2.20.0
    
    echo "✓ Core dependencies installed"
fi
echo ""

# Verify installation
echo "Step 6: Verifying installation..."
echo ""

# Test PyTorch
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "ERROR: PyTorch import failed!"
    exit 1
}

# Test CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "WARNING: CUDA check failed (CPU-only mode)"
}

# Test audio libraries
python3 -c "import soundfile; print(f'soundfile version: {soundfile.__version__}')" || {
    echo "ERROR: soundfile import failed!"
    exit 1
}

# Test metrics
python3 -c "from pesq import pesq; print('PESQ library: OK')" || {
    echo "ERROR: PESQ import failed!"
    exit 1
}

python3 -c "from pystoi import stoi; print('PYSTOI library: OK')" || {
    echo "ERROR: PYSTOI import failed!"
    exit 1
}

echo ""
echo "✓ All critical libraries verified"
echo ""

# Create TMPDIR if needed
TMPDIR_PATH="/mnt/Data/muaw1874/tmp"
if [ ! -d "$TMPDIR_PATH" ]; then
    echo "Step 7: Creating TMPDIR..."
    mkdir -p "$TMPDIR_PATH"
    echo "✓ Created $TMPDIR_PATH"
else
    echo "Step 7: TMPDIR already exists at $TMPDIR_PATH"
fi
echo ""

# Print summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment: $VENV_NAME"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To verify the best model:"
echo "  $VENV_NAME/bin/python scripts/evaluate_emergency.py \\"
echo "    --ckpt checkpoints_emergency/best_pesq_finetune.pt \\"
echo "    --out eval_final_comprehensive.txt \\"
echo "    --n-files 20"
echo ""
echo "Next steps:"
echo "  1. Read conversation_context.md for complete project history"
echo "  2. Read PROMPT_FOR_NEW_SERVER.md for resumption instructions"
echo "  3. Run the evaluation command above to verify best model"
echo ""
echo "=========================================="
