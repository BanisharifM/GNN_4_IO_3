#!/bin/bash
#SBATCH --job-name=preprocess_adv_feat
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/preprocess_adv_feat_%j.out
#SBATCH --error=logs/preprocess_adv_feat_%j.err

# Load required modules
module load anaconda3

# Activate conda environment (replace with your environment name)
source activate gnn-env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables
DATA_DIR="data/split_data/sample_100"
OUTPUT_BASE_DIR="data/preprocessed/adv_feat_gcn/experiment_2"
MI_FILE="data/mutual_information2.csv"
MI_THRESHOLD=0.3259
TOP_FEATURES=15

# Create output directories
mkdir -p ${OUTPUT_BASE_DIR}/{train,val,test}

echo "Starting advanced feature selection preprocessing..."

# Process training data
echo "Preprocessing train data from ${DATA_DIR}/train.csv..."
python 01_preprocess_data_paper_based.py \
  --data_file ${DATA_DIR}/train.csv \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_BASE_DIR}/train \
  --mi_threshold ${MI_THRESHOLD} \
  --use_advanced_feature_selection True \
  --top_features ${TOP_FEATURES} \
  --split_type train

# Process validation data
echo "Preprocessing val data from ${DATA_DIR}/val.csv..."
python 01_preprocess_data_paper_based.py \
  --data_file ${DATA_DIR}/val.csv \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_BASE_DIR}/val \
  --mi_threshold ${MI_THRESHOLD} \
  --use_advanced_feature_selection True \
  --top_features ${TOP_FEATURES} \
  --split_type val

# Process test data
echo "Preprocessing test data from ${DATA_DIR}/test.csv..."
python 01_preprocess_data_paper_based.py \
  --data_file ${DATA_DIR}/test.csv \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_BASE_DIR}/test \
  --mi_threshold ${MI_THRESHOLD} \
  --use_advanced_feature_selection True \
  --top_features ${TOP_FEATURES} \
  --split_type test

echo "Preprocessing completed. Results saved to ${OUTPUT_BASE_DIR}/"
