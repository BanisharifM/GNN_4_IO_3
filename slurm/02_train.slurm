#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4-interactive 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/experiment_2/train_%j.out
#SBATCH --error=logs/slurm/experiment_2/train_%j.err

# Load required modules
# module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables for split-based workflow
TRAIN_DIR="data/preprocessed/baseline_gcn/experiment_2/train"
VAL_DIR="data/preprocessed/baseline_gcn/experiment_2/val"
TEST_DIR="data/preprocessed/baseline_gcn/experiment_2/test"
OUTPUT_DIR="logs/training/baseline_gcn/experiment_2"

# Run training script with single checkpoint
echo "Starting model training at $(date)"
python scripts/02_train_model.py \
  --train_dir ${TRAIN_DIR} \
  --val_dir ${VAL_DIR} \
  --test_dir ${TEST_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type gcn \
  --hidden_dim 64 \
  --num_layers 2 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 20 \
  --dropout 0.1 \
  --early_stopping_patience 10 \
  --device cuda \
  --use_split_dirs True \
  --resume_training True

echo "Training completed at $(date)"
