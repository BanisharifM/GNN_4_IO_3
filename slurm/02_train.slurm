#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load required modules
module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables
PREPROCESSED_DIR="/path/to/preprocessed"
OUTPUT_DIR="/path/to/model"

# Run training script
echo "Starting model training at $(date)"
python scripts/02_train_model.py \
  --preprocessed_dir ${PREPROCESSED_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --hidden_dim 64 \
  --num_layers 2 \
  --model_type gcn \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 100 \
  --dropout 0.1 \
  --early_stopping_patience 10 \
  --device cuda

echo "Training completed at $(date)"
