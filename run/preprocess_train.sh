#!/bin/bash
# Simple script to preprocess data for baseline GCN model

# Default parameters
DATA_FILE="data/split_data/sample_100/train.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR="data/preprocessed/baseline_gcn/train"
SPLIT_TYPE="train"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the preprocess data script
echo "Preprocessing ${SPLIT_TYPE} data from ${DATA_FILE}..."
python  -W ignore scripts/01_preprocess_data.py \
    --data_file ${DATA_FILE} \
    --mi_file ${MI_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --split_type ${SPLIT_TYPE} \
    --use_advanced_feature_selection False \
    --use_clustering False

echo "Preprocessing completed. Results saved to ${OUTPUT_DIR}/"
