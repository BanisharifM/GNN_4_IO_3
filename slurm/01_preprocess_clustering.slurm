#!/bin/bash
#SBATCH --job-name=preprocess_clustering
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/experiment_6/preprocess_clustering_%j.out
#SBATCH --error=logs/slurm/experiment_6/preprocess_clustering_%j.err

# Load modules
# module load anaconda3
# module load cuda  # Uncomment if CUDA-dependent libs are used

# Activate environment
source activate gnn-env

# Create logs directory
mkdir -p logs

# Parameters
MI_THRESHOLD=0.3259
NUM_CLUSTERS=4
CONFIG_TAG="mi${MI_THRESHOLD}_k${NUM_CLUSTERS}"

# Paths
BASE_DATA_DIR="data/split_data/sample_total"
MI_FILE="data/mutual_information2.csv"
OUTPUT_BASE_DIR="data/preprocessed/clustering/${CONFIG_TAG}"

# Create output directories
for split in train val test; do
    mkdir -p ${OUTPUT_BASE_DIR}/${split}
done

# Run preprocessing with clustering
for split in train val test; do
    echo "Preprocessing ${split} data with clustering..."
    python scripts/01_preprocess_data.py \
      --data_file ${BASE_DATA_DIR}/${split}.csv \
      --mi_file ${MI_FILE} \
      --output_dir ${OUTPUT_BASE_DIR}/${split} \
      --mi_threshold ${MI_THRESHOLD} \
      --use_clustering True \
      --num_clusters ${NUM_CLUSTERS} \
      --split_type ${split}
done

echo "Clustering-based preprocessing completed at $(date)"
