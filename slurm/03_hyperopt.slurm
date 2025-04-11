#!/bin/bash
#SBATCH --job-name=gnn_hyperopt
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/experiment_6/hyperopt_%j.out
#SBATCH --error=logs/slurm/experiment_6/hyperopt_%j.err

# Load required modules
module load cuda

# Activate conda environment
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs/slurm/experiment_6

# Set variables for split-based workflow
TRAIN_DIR="data/preprocessed/baseline_gcn/experiment_3/train"
VAL_DIR="data/preprocessed/baseline_gcn/experiment_3/val"
TEST_DIR="data/preprocessed/baseline_gcn/experiment_3/test"
OUTPUT_DIR="logs/hyperopt/baseline_gcn/experiment_6"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run hyperparameter optimization script
echo "Starting hyperparameter optimization at $(date)"
python scripts/04_hyperparameter_optimization.py \
  --train_dir ${TRAIN_DIR} \
  --val_dir ${VAL_DIR} \
  --test_dir ${TEST_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --num_samples 40 \
  --max_epochs 10 \
  --early_stopping_patience 5 \
  --cpus_per_trial 4 \
  --gpus_per_trial 0.5 \
  --num_workers 4

echo "Hyperparameter optimization completed at $(date)"
echo "TensorBoard logs available at: ${OUTPUT_DIR}/tensorboard"
echo "To view TensorBoard logs, run: tensorboard --logdir=${OUTPUT_DIR}/tensorboard"
