#!/bin/bash
#SBATCH --job-name=tadiff_test
#SBATCH --output=logs/tadiff/output.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node  
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=0:03:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

source /apps/local/anaconda3/conda_init.sh
conda activate wm    
srun python -m scripts.test --patient_ids 32 --diffusion_steps 50 --num_samples 4
