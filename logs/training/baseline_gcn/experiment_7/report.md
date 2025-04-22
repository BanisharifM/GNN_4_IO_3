#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/experiment_7/train_%j.out
#SBATCH --error=logs/slurm/experiment_7/train_%j.err

# Load required modules
# module load anaconda3
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Set variables for split-based workflow
TRAIN_DIR="data/preprocessed/baseline/mi0.2/train"
VAL_DIR="data/preprocessed/baseline/mi0.2/val"
TEST_DIR="data/preprocessed/baseline/mi0.2/test"
OUTPUT_DIR="logs/training/baseline_gcn/experiment_7"

# Run training script with single checkpoint
echo "Starting model training at $(date)"
python scripts/02_train_model.py \
  --train_dir ${TRAIN_DIR} \
  --val_dir ${VAL_DIR} \
  --test_dir ${TEST_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type gcn \
  --hidden_dim 256 \
  --num_layers 2 \
  --learning_rate 0.0017331607338165434 \
  --batch_size 64 \
  --epochs 20 \
  --dropout 0.23487409763750228 \
  --early_stopping_patience 10 \
  --device cuda \
  --use_split_dirs
  # --no_resume If want to disable resume (start fresh)


echo "Training completed at $(date)"



2025-04-21 17:51:04,553 - INFO - Using device: cuda
2025-04-21 17:51:04,553 - INFO - Using split-based workflow with directories: data/preprocessed/baseline/mi0.2/train, data/preprocessed/baseline/mi0.2/val, data/preprocessed/baseline/mi0.2/test
2025-04-21 17:51:05,503 - INFO - Creating datasets from split directories...
2025-04-21 17:54:39,043 - INFO - Train/Val/Test sizes: 4653053/997082/997084
2025-04-21 17:54:39,043 - INFO - Creating GCN model with 2 layers and 256 hidden dimensions
2025-04-21 17:54:39,144 - INFO - Initialized GCN model with 2 layers and 256 hidden dimensions
2025-04-21 17:54:39,145 - INFO - Initialized GNN regressor with GCN backbone
2025-04-21 17:54:39,623 - INFO - No checkpoint found or resume disabled. Starting training from scratch.
2025-04-21 17:54:39,624 - INFO - Starting training for 20 epochs (from epoch 1)
2025-04-21 18:04:02,560 - INFO - Epoch 1/20, Train Loss: 310039589.8814, Val Loss: 0.9490
2025-04-21 18:04:02,584 - INFO - New best model saved at epoch 1 with validation loss: 0.949009
2025-04-21 18:13:36,935 - INFO - Epoch 2/20, Train Loss: 0.9496, Val Loss: 0.9490
2025-04-21 18:13:36,950 - INFO - No improvement for 1 epochs. Best validation loss: 0.949009 at epoch 1
2025-04-21 18:23:14,536 - INFO - Epoch 3/20, Train Loss: 0.9496, Val Loss: 0.9489
2025-04-21 18:23:14,562 - INFO - New best model saved at epoch 3 with validation loss: 0.948919
2025-04-21 18:32:33,414 - INFO - Epoch 4/20, Train Loss: 0.9495, Val Loss: 0.9492
2025-04-21 18:32:33,428 - INFO - No improvement for 1 epochs. Best validation loss: 0.948919 at epoch 3
2025-04-21 18:41:53,651 - INFO - Epoch 5/20, Train Loss: 0.9496, Val Loss: 0.9490
2025-04-21 18:41:53,666 - INFO - No improvement for 2 epochs. Best validation loss: 0.948919 at epoch 3
slurmstepd: error: *** JOB 9274167 ON gpua023 CANCELLED AT 2025-04-21T18:51:06 DUE TO TIME LIMIT ***




