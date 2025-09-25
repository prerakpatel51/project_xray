#!/bin/bash
#SBATCH --job-name=jupyter-gpu
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=jupyter-%j.out

# Load any required modules
# module load python/3.9 cuda/11.8  # Uncomment and adjust as needed

# Start Jupyter
cd /home1/ppatel2025/project_xray
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root