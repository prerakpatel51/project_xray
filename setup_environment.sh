#!/bin/bash

# Setup script for Vision Mamba environment
echo "Setting up Vision Mamba environment..."

# Create and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision_mamba

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install compilation tools first
echo "Installing compilation dependencies..."
pip install packaging ninja wheel setuptools

# Install core dependencies
echo "Installing core dependencies..."
pip install timm einops transformers

# Install remaining dependencies first
echo "Installing remaining packages..."
pip install numpy pandas Pillow opencv-python matplotlib seaborn scikit-learn tqdm wandb albumentations

# Install Mamba dependencies last (might take time to compile)
echo "Installing Mamba SSM (this may take several minutes)..."
export MAX_JOBS=4  # Limit parallel compilation jobs
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=1.2.0

# Test installation
echo "Testing installation..."
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "try: import mamba_ssm; print('Mamba-SSM: OK'); except: print('Mamba-SSM: FALLBACK to MultiHeadAttention')"

echo "Environment setup complete!"
echo "To activate: conda activate vision_mamba"