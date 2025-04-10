#!/bin/bash
# Simple script to preprocess data for test set with advanced feature selection

mkdir -p data/preprocessed/adv_feat/total/train
mkdir -p data/preprocessed/adv_feat/total/val
mkdir -p data/preprocessed/adv_feat/total/test

# Set variables
TRAIN_DATA="data/split_data/sample_total/train.csv"
VAL_DATA="data/split_data/sample_total/val.csv"
TEST_DATA="data/split_data/sample_total/test.csv"
MI_FILE="data/mutual_information2.csv"
OUTPUT_DIR_TRAIN="data/preprocessed/adv_feat/total/train"
OUTPUT_DIR_VAL="data/preprocessed/adv_feat/total/val"
OUTPUT_DIR_TEST="data/preprocessed/adv_feat/total/test"

# Run preprocessing with advanced feature selection for train data
echo "Preprocessing train data with advanced feature selection..."
python scripts/01_preprocess_data.py \
  --data_file ${TRAIN_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_TRAIN} \
  --split_type train \
  --use_advanced_feature_selection True

# Run preprocessing with advanced feature selection for validation data
echo "Preprocessing validation data with advanced feature selection..."
python scripts/01_preprocess_data.py \
  --data_file ${VAL_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_VAL} \
  --split_type val \
  --use_advanced_feature_selection True

# Run preprocessing with advanced feature selection for test data
echo "Preprocessing test data with advanced feature selection..."
python scripts/01_preprocess_data.py \
  --data_file ${TEST_DATA} \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR_TEST} \
  --split_type test \
  --use_advanced_feature_selection True

echo "Advanced feature selection preprocessing completed at $(date)"