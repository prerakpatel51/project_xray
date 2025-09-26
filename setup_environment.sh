#!/bin/bash

# Setup script for Vision Mamba environment
echo "Setting up Vision Mamba environment..."

# Create and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision_mamba



pip install torch
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/accelerate
pip install huggingface_hub

git clone https://github.com/state-spaces/mamba.git && cd mamba
CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE pip install .

pip install git+https://github.com/Dao-AILab/causal-conv1d

# Test installation
echo "Testing installation..."
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('Mamba-SSM: OK')"

echo "Environment setup complete!"
echo "To activate: conda activate vision_mamba"