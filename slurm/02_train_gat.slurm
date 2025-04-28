#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm/experiment_13/train_%j.out
#SBATCH --error=logs/slurm/experiment_13/train_%j.err

# Load required modules
# module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables for split-based workflow
TRAIN_DIR="data/preprocessed/baseline/mi0.1/train"
VAL_DIR="data/preprocessed/baseline/mi0.1/val"
TEST_DIR="data/preprocessed/baseline/mi0.1/test"
OUTPUT_DIR="logs/training/baseline_gat/experiment_13"

# Run training script with single checkpoint
echo "Starting model training at $(date)"
python scripts/02_train_model.py \
  --train_dir ${TRAIN_DIR} \
  --val_dir ${VAL_DIR} \
  --test_dir ${TEST_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type gat \
  --hidden_dim 128 \
  --num_layers 3 \
  --learning_rate 0.0007899744253712328 \
  --batch_size 16 \
  --epochs 50 \
  --dropout 0.18051110873398107 \
  --early_stopping_patience 20 \
  --device cuda \
  --use_split_dirs
  # --no_resume If want to disable resume (start fresh)

echo "Training completed at $(date)"
