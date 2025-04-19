#!/bin/bash
#SBATCH --job-name=preprocess_hybrid
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/experiment_6/preprocess_hybrid_%j.out
#SBATCH --error=logs/slurm/experiment_6/preprocess_hybrid_%j.err

# Load modules
# module load anaconda3
# module load cuda  # Uncomment if needed for GPU-related packages

# Activate conda environment
source activate gnn-env

# Create logs directory
mkdir -p logs

# Parameters
MI_THRESHOLD=0.3259
TOP_FEATURES=15
NUM_CLUSTERS=4
CONFIG_TAG="mi${MI_THRESHOLD}_top${TOP_FEATURES}_k${NUM_CLUSTERS}"

# Paths
BASE_DATA_DIR="data/split_data/sample_total"
MI_FILE="data/mutual_information2.csv"
OUTPUT_BASE_DIR="data/preprocessed/hybrid/${CONFIG_TAG}"

# Create output directories
for split in train val test; do
    mkdir -p ${OUTPUT_BASE_DIR}/${split}
done

# Run preprocessing with hybrid config (feature selection + clustering)
for split in train val test; do
    echo "Preprocessing ${split} data with hybrid method (feature selection + clustering)..."
    python scripts/01_preprocess_data.py \
      --data_file ${BASE_DATA_DIR}/${split}.csv \
      --mi_file ${MI_FILE} \
      --output_dir ${OUTPUT_BASE_DIR}/${split} \
      --mi_threshold ${MI_THRESHOLD} \
      --use_advanced_feature_selection True \
      --top_features ${TOP_FEATURES} \
      --use_clustering True \
      --num_clusters ${NUM_CLUSTERS} \
      --split_type ${split}
done

echo "Hybrid preprocessing completed at $(date)"
