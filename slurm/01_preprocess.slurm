#!/bin/bash
#SBATCH --job-name=preprocess_baseline
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/experiment_7/preprocess_baseline_%j.out
#SBATCH --error=logs/slurm/experiment_7/preprocess_baseline_%j.err

# Activate your environment
source activate gnn-env

# Logging
mkdir -p logs

# Configurable parameters
MI_THRESHOLD=0.2
CONFIG_TAG="mi${MI_THRESHOLD}"  

# Paths
BASE_DATA_DIR="data/split_data/sample_total"
MI_FILE="data/mutual_information2.csv"
OUTPUT_BASE_DIR="data/preprocessed/baseline/${CONFIG_TAG}"

# Create output directories
for split in train val test; do
    mkdir -p ${OUTPUT_BASE_DIR}/${split}
done

# Run preprocessing for each split
for split in train val test; do
    echo "Preprocessing ${split} data for baseline (mi_threshold=${MI_THRESHOLD})..."
    python scripts/01_preprocess_data.py \
      --data_file ${BASE_DATA_DIR}/${split}.csv \
      --mi_file ${MI_FILE} \
      --output_dir ${OUTPUT_BASE_DIR}/${split} \
      --mi_threshold ${MI_THRESHOLD} \
      --use_advanced_feature_selection False \
      --use_clustering False \
      --split_type ${split}
done

echo "Baseline preprocessing completed at $(date)"
