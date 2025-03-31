#!/bin/bash
#SBATCH --job-name=gnn_preprocess
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Load required modules
module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables
DATA_DIR="/path/to/data"
OUTPUT_DIR="/path/to/preprocessed"
SAMPLE_SIZE=1000  # Set to None for full dataset or a number for testing

# Run preprocessing script
echo "Starting preprocessing at $(date)"
python scripts/01_preprocess_data.py \
  --data_file ${DATA_DIR}/sample_train_100.csv \
  --mi_file ${DATA_DIR}/mutual_information2.csv \
  --output_dir ${OUTPUT_DIR} \
  --mi_threshold 0.3259 \
  --sample_size ${SAMPLE_SIZE}

echo "Preprocessing completed at $(date)"
