#!/bin/bash
#SBATCH --job-name=train_gcn_clustering
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_gcn_clustering_%j.out
#SBATCH --error=logs/train_gcn_clustering_%j.err

# Load required modules
module load anaconda3
module load cuda/11.7

# Activate conda environment (replace with your environment name)
source activate gnn-env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables
PREPROCESSED_DIR="data/preprocessed/hybrid_gcn/experiment_2"
OUTPUT_DIR="logs/training/gcn_clustering"
HIDDEN_DIM=64
NUM_LAYERS=2
MODEL_TYPE="gcn"
LEARNING_RATE=0.001
BATCH_SIZE=32
EPOCHS=100
DROPOUT=0.1
EARLY_STOPPING=10

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Starting GCN training with clustering information..."

# Run training with clustering as feature
python 02_train_model_with_clustering.py \
  --train_dir ${PREPROCESSED_DIR}/train \
  --val_dir ${PREPROCESSED_DIR}/val \
  --test_dir ${PREPROCESSED_DIR}/test \
  --output_dir ${OUTPUT_DIR}/cluster_as_feature \
  --use_split_dirs \
  --hidden_dim ${HIDDEN_DIM} \
  --num_layers ${NUM_LAYERS} \
  --model_type ${MODEL_TYPE} \
  --learning_rate ${LEARNING_RATE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --dropout ${DROPOUT} \
  --early_stopping_patience ${EARLY_STOPPING} \
  --use_clustering \
  --cluster_as_feature

echo "Training with cluster as feature completed. Results saved to ${OUTPUT_DIR}/cluster_as_feature/"

# Run training with separate models per cluster
python 02_train_model_with_clustering.py \
  --train_dir ${PREPROCESSED_DIR}/train \
  --val_dir ${PREPROCESSED_DIR}/val \
  --test_dir ${PREPROCESSED_DIR}/test \
  --output_dir ${OUTPUT_DIR}/per_cluster_models \
  --use_split_dirs \
  --hidden_dim ${HIDDEN_DIM} \
  --num_layers ${NUM_LAYERS} \
  --model_type ${MODEL_TYPE} \
  --learning_rate ${LEARNING_RATE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --dropout ${DROPOUT} \
  --early_stopping_patience ${EARLY_STOPPING} \
  --use_clustering \
  --train_per_cluster

echo "Training with separate models per cluster completed. Results saved to ${OUTPUT_DIR}/per_cluster_models/"
