#!/bin/bash
#SBATCH --job-name=gnn_shap
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/shap_%j.out
#SBATCH --error=logs/shap_%j.err

# Load required modules
module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables
PREPROCESSED_DIR="/path/to/preprocessed"
MODEL_DIR="/path/to/model"
OUTPUT_DIR="/path/to/shap_results"

# Run SHAP analysis script
echo "Starting SHAP analysis at $(date)"
python scripts/03_analyze_bottlenecks.py \
  --preprocessed_dir ${PREPROCESSED_DIR} \
  --model_dir ${MODEL_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --background_samples 50 \
  --test_samples 20 \
  --top_n 10 \
  --device cuda

echo "SHAP analysis completed at $(date)"
