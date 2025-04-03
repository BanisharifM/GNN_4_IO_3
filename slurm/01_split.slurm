#!/bin/bash
#SBATCH --job-name=GNN_Split
#SBATCH --output=logs/slurm/split_%j.out
#SBATCH --error=logs/slurm/split_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bdau-delta-gpu

# === Pre-run Setup ===
mkdir -p logs/slurm
echo "SLURM job started at $(date)"

# === Load environment ===
module load anaconda3
module load cuda

# === Activate your conda environment ===
source activate gnn_env

# === Set variables ===
DATA_FILE="data/sample_train_total.csv"
OUTPUT_DIR="data/split_data/sample_100000"
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15
SAMPLE_SIZE=100000

# === Create output directory ===
mkdir -p ${OUTPUT_DIR}

# === Run the split script ===
echo "Splitting data from ${DATA_FILE} into train/val/test sets..."
python scripts/00_split_data.py \
  --data_file ${DATA_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --train_ratio ${TRAIN_RATIO} \
  --val_ratio ${VAL_RATIO} \
  --test_ratio ${TEST_RATIO} \
  --sample_size ${SAMPLE_SIZE}

# === Job End ===
echo "Data splitting completed at $(date)"
