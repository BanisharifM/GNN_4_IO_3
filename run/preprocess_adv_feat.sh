#!/bin/bash
# Simple script to preprocess data for test set with advanced feature selection

# TEST SET
# Default parameters
DATA_FILE="data/split_data/sample_total/test.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR="data/preprocessed/adv_feat/total/test"
SPLIT_TYPE="test"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the preprocess data script
echo "Preprocessing ${SPLIT_TYPE} data from ${DATA_FILE}..."
python scripts/01_preprocess_data.py \
    --data_file ${DATA_FILE} \
    --mi_file ${MI_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --split_type ${SPLIT_TYPE} \
    --use_advanced_feature_selection False \
    --use_clustering False \
    --top_features 10

echo "Preprocessing completed. Results saved to ${OUTPUT_DIR}/"


#TRAINING SET
# Default parameters
DATA_FILE="data/split_data/sample_total/train.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR="data/preprocessed/adv_feat/total/train"
SPLIT_TYPE="train"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the preprocess data script
echo "Preprocessing ${SPLIT_TYPE} data from ${DATA_FILE}..."
python scripts/01_preprocess_data.py \
    --data_file ${DATA_FILE} \
    --mi_file ${MI_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --split_type ${SPLIT_TYPE} \
    --use_advanced_feature_selection False \
    --use_clustering False \
    --top_features 10

echo "Preprocessing completed. Results saved to ${OUTPUT_DIR}/"

#VALIDATION SET
# Default parameters
DATA_FILE="data/split_data/sample_total/val.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR="data/preprocessed/adv_feat/total/val"
SPLIT_TYPE="val"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the preprocess data script
echo "Preprocessing ${SPLIT_TYPE} data from ${DATA_FILE}..."
python scripts/01_preprocess_data.py \
    --data_file ${DATA_FILE} \
    --mi_file ${MI_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --split_type ${SPLIT_TYPE} \
    --use_advanced_feature_selection False \
    --use_clustering False \
    --top_features 10

echo "Preprocessing completed. Results saved to ${OUTPUT_DIR}/"