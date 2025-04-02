#!/bin/bash
# Simple script to train GCN model with split-based workflow

# Default parameters
TRAIN_DIR="data/preprocessed/baseline_gcn/train"
VAL_DIR="data/preprocessed/baseline_gcn/val"
TEST_DIR="data/preprocessed/baseline_gcn/test"
OUTPUT_DIR="logs/training/baseline_gcn"
MODEL_TYPE="gcn"
HIDDEN_DIM=64
NUM_LAYERS=2

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the modified train model script
echo "Training ${MODEL_TYPE} model using split data..."
python -W ignore scripts/02_train_model.py \
    --train_dir ${TRAIN_DIR} \
    --val_dir ${VAL_DIR} \
    --test_dir ${TEST_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --use_split_dirs True

echo "Training completed. Model saved to ${OUTPUT_DIR}/"
