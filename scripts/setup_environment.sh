#!/bin/bash

# Setup script for GAN-LUAD Clustering Project
# Phase 1: Environment Setup

echo "======================================================================"
echo "GAN-LUAD Clustering - Environment Setup"
echo "======================================================================"

# Check Python version
echo -e "\n[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if Python 3.8+ is installed
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✓ Python version is compatible (3.8+)"
else
    echo "✗ Error: Python 3.8+ is required"
    exit 1
fi

# Create virtual environment
echo -e "\n[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n): " recreate
    if [ "$recreate" = "y" ]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[3/6] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo -e "\n[4/6] Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Install PyTorch with CUDA support (if available)
echo -e "\n[5/6] Installing PyTorch..."
echo "Detecting CUDA availability..."

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "✓ CUDA Version: $cuda_version"
    
    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
else
    echo "⚠ No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision
fi

echo "✓ PyTorch installed"

# Install other dependencies
echo -e "\n[6/6] Installing other dependencies..."
pip install -r backend/requirements.txt
echo "✓ All dependencies installed"

# Create .env file from example
echo -e "\n[Setup] Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env file created from .env.example"
    echo "⚠ Please update .env with your specific settings if needed"
else
    echo "✓ .env file already exists"
fi

# Create directory structure
echo -e "\n[Setup] Creating directory structure..."
python3 -c "from backend.config import *"
echo "✓ Directory structure created"

# Test installation
echo -e "\n======================================================================"
echo "Testing Installation"
echo "======================================================================"

python3 << END
import sys
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", seaborn.__version__)

print("\n✓ All core packages imported successfully!")
END

# Final instructions
echo -e "\n======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Download the data:"
echo "   python scripts/download_data.py"
echo ""
echo "2. Run the data exploration notebook:"
echo "   jupyter notebook notebooks/01_data_exploration.ipynb"
echo ""
echo "3. Or activate the virtual environment manually:"
echo "   source venv/bin/activate"
echo ""
echo "======================================================================"