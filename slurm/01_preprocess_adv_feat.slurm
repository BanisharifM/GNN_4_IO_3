#!/bin/bash
#SBATCH --job-name=preprocess_adv_feat
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/experiment_6/preprocess_adv_feat_%j.out
#SBATCH --error=logs/slurm/experiment_6/preprocess_adv_feat_%j.err

# Load modules
# module load anaconda3
# module load cuda  # If needed for GPU-accelerated environments

# Activate environment
source activate gnn-env

# Create logs directory
mkdir -p logs

# Parameters
MI_THRESHOLD=0.3259
TOP_FEATURES=15
CONFIG_TAG="mi${MI_THRESHOLD}_top${TOP_FEATURES}"

# Paths
BASE_DATA_DIR="data/split_data/sample_total"
MI_FILE="data/mutual_information2.csv"
OUTPUT_BASE_DIR="data/preprocessed/adv_feat/${CONFIG_TAG}"

# Create output directories
for split in train val test; do
    mkdir -p ${OUTPUT_BASE_DIR}/${split}
done

# Run preprocessing with advanced feature selection
for split in train val test; do
    echo "Preprocessing ${split} data with advanced feature selection..."
    python scripts/01_preprocess_data.py \
      --data_file ${BASE_DATA_DIR}/${split}.csv \
      --mi_file ${MI_FILE} \
      --output_dir ${OUTPUT_BASE_DIR}/${split} \
      --use_advanced_feature_selection True \
      --mi_threshold ${MI_THRESHOLD} \
      --top_features ${TOP_FEATURES} \
      --split_type ${split}
done

echo "Advanced feature selection preprocessing completed at $(date)"
