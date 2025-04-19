#!/bin/bash
# Simple script to preprocess data with clustering-based approach

# Create output directories
mkdir -p data/preprocessed/clustering/train
mkdir -p data/preprocessed/clustering/val
mkdir -p data/preprocessed/clustering/test

# Set variables
TRAIN_DATA="data/split_data/sample_total/train.csv"
VAL_DATA="data/split_data/sample_total/val.csv"
TEST_DATA="data/split_data/sample_total/test.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR_TRAIN="data/preprocessed/clustering/train"
OUTPUT_DIR_VAL="data/preprocessed/clustering/val"
OUTPUT_DIR_TEST="data/preprocessed/clustering/test"
MI_THRESHOLD=0.3259
NUM_CLUSTERS=4

# Run preprocessing with clustering for train data
echo "Preprocessing train data with clustering..."
python scripts/01_preprocess_data.py \
  --data_file ${TRAIN_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_TRAIN} \
  --mi_threshold ${MI_THRESHOLD} \
  --use_clustering True \
  --num_clusters ${NUM_CLUSTERS} \
  --split_type train

# Run preprocessing with clustering for validation data
echo "Preprocessing validation data with clustering..."
python scripts/01_preprocess_data.py \
  --data_file ${VAL_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_VAL} \
  --mi_threshold ${MI_THRESHOLD} \
  --use_clustering True \
  --num_clusters ${NUM_CLUSTERS} \
  --split_type val

# Run preprocessing with clustering for test data
echo "Preprocessing test data with clustering..."
python scripts/01_preprocess_data.py \
  --data_file ${TEST_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_TEST} \
  --mi_threshold ${MI_THRESHOLD} \
  --use_clustering True \
  --num_clusters ${NUM_CLUSTERS} \
  --split_type test

echo "Clustering-based preprocessing completed at $(date)"
