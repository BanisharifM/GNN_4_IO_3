#!/bin/bash
# Simple script to train GCN model

# Default parameters
PREPROCESSED_DIR="data/preprocessed/baseline_gcn"
OUTPUT_DIR="logs/training/baseline_gcn"
MODEL_TYPE="gcn"
HIDDEN_DIM=64
NUM_LAYERS=2

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the train model script
echo "Training ${MODEL_TYPE} model using data from ${PREPROCESSED_DIR}..."
python scripts/02_train_model.py \
    --preprocessed_dir ${PREPROCESSED_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS}

echo "Training completed. Model saved to ${OUTPUT_DIR}/"
