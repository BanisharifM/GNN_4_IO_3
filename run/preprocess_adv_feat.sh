#!/bin/bash
# Simple script to preprocess data for all dataset with advanced feature selection

mkdir -p data/preprocessed/adv_feat/train
mkdir -p data/preprocessed/adv_feat/val
mkdir -p data/preprocessed/adv_feat/test

# Set variables
TRAIN_DATA="data/split_data/sample_total/train.csv"
VAL_DATA="data/split_data/sample_total/val.csv"
TEST_DATA="data/split_data/sample_total/test.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR_TRAIN="data/preprocessed/adv_feat/train"
OUTPUT_DIR_VAL="data/preprocessed/adv_feat/val"
OUTPUT_DIR_TEST="data/preprocessed/adv_feat/test"
MI_THRESHOLD=0.3259
TOP_FEATURES=15

# Run preprocessing with advanced feature selection for train data
echo "Preprocessing train data with advanced feature selection..."
python scripts/01_preprocess_data.py \
  --data_file ${TRAIN_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_TRAIN} \
  --use_advanced_feature_selection True \
  --mi_threshold ${MI_THRESHOLD} \
  --top_features ${TOP_FEATURES} \
  --split_type train

# Run preprocessing with advanced feature selection for validation data
echo "Preprocessing validation data with advanced feature selection..."
python scripts/01_preprocess_data.py \
  --data_file ${VAL_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_VAL} \
  --use_advanced_feature_selection True \
  --mi_threshold ${MI_THRESHOLD} \
  --top_features ${TOP_FEATURES} \
  --split_type val

# Run preprocessing with advanced feature selection for test data
echo "Preprocessing test data with advanced feature selection..."
python scripts/01_preprocess_data.py \
  --data_file ${TEST_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_TEST} \
  --use_advanced_feature_selection True \
  --mi_threshold ${MI_THRESHOLD} \
  --top_features ${TOP_FEATURES} \
  --split_type test

echo "Advanced feature selection preprocessing completed at $(date)"
