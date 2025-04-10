#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/experiment_5/train_%j.out
#SBATCH --error=logs/slurm/experiment_5/train_%j.err

# Load required modules
# module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables for split-based workflow
TRAIN_DIR="data/preprocessed/baseline_gcn/experiment_3/train"
VAL_DIR="data/preprocessed/baseline_gcn/experiment_3/val"
TEST_DIR="data/preprocessed/baseline_gcn/experiment_3/test"
OUTPUT_DIR="logs/training/baseline_gat/experiment_5"

# Run training script with single checkpoint
echo "Starting model training at $(date)"
python scripts/02_train_model.py \
  --train_dir ${TRAIN_DIR} \
  --val_dir ${VAL_DIR} \
  --test_dir ${TEST_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type gat \
  --hidden_dim 256 \
  --num_layers 2 \
  --learning_rate 0.0017331607338165434 \
  --batch_size 64 \
  --epochs 10 \
  --dropout 0.23487409763750228 \
  --early_stopping_patience 10 \
  --device cuda \
  --use_split_dirs True \
  --resume_training True

echo "Training completed at $(date)"
