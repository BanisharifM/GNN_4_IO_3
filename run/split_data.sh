#!/bin/bash
# Simple script to split data into train/val/test sets

# Default parameters
DATA_FILE="data/sample_train_total.csv"
OUTPUT_DIR="data/split_data/sample_1,000,000"
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15
SAMPLE_SIZE=1000000


# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the split data script
echo "Splitting data from ${DATA_FILE} into train/val/test sets..."
python scripts/00_split_data.py \
    --data_file ${DATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --train_ratio ${TRAIN_RATIO} \
    --val_ratio ${VAL_RATIO} \
    --test_ratio ${TEST_RATIO} \
    --sample_size ${SAMPLE_SIZE}

echo "Data splitting completed. Files saved to ${OUTPUT_DIR}/"
