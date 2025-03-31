#!/bin/bash
#SBATCH --job-name=gnn_hyperopt
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hyperopt_%j.out
#SBATCH --error=logs/hyperopt_%j.err

# Load required modules
module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables
PREPROCESSED_DIR="/path/to/preprocessed"
OUTPUT_DIR="/path/to/hyperopt_results"

# Run hyperparameter optimization script
echo "Starting hyperparameter optimization at $(date)"
python scripts/04_hyperparameter_optimization.py \
  --preprocessed_dir ${PREPROCESSED_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --num_samples 20 \
  --max_epochs 100 \
  --early_stopping_patience 10 \
  --cpus_per_trial 4 \
  --gpus_per_trial 0.5 \
  --num_workers 4

echo "Hyperparameter optimization completed at $(date)"
