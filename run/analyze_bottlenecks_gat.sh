#!/bin/bash
# Simple script to analyze bottlenecks using SHAP for GAT model

# Default parameters
PREPROCESSED_DIR="data/preprocessed/baseline_gat"
MODEL_DIR="logs/training/baseline_gat"
OUTPUT_DIR="logs/shap/baseline_gat"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run the bottleneck analysis script
echo "Analyzing bottlenecks using SHAP for model in ${MODEL_DIR}..."
python scripts/03_analyze_bottlenecks.py \
    --preprocessed_dir ${PREPROCESSED_DIR} \
    --model_dir ${MODEL_DIR} \
    --output_dir ${OUTPUT_DIR}

echo "Bottleneck analysis completed. Results saved to ${OUTPUT_DIR}/"
