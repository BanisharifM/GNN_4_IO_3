#!/bin/bash
#SBATCH --job-name=GNN_Training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:1
#SBATCH --account=bdau-delta-gpu

# Load necessary modules
module load python/3.11.6
module load cuda/11.8

# Activate virtual environment (replace with your environment path)
source /path/to/your/venv/bin/activate

# Create experiment directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="experiments/train_${TIMESTAMP}"
mkdir -p ${EXPERIMENT_DIR}/logs
mkdir -p ${EXPERIMENT_DIR}/checkpoints

# Set environment variables for distributed training
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=localhost  # Use localhost for single node
export MASTER_PORT=29511      # Use a different, unused port
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export OMP_NUM_THREADS=4

# NCCL settings
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

# Debugging output
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"

# Set variables
PREPROCESSED_DIR="experiments/latest_preprocess/preprocessed"  # Use latest preprocessed data
OUTPUT_DIR="${EXPERIMENT_DIR}/model"
RESUME_CHECKPOINT=""  # Leave empty for new training or specify path to resume

# Model hyperparameters
HIDDEN_DIM=64
NUM_LAYERS=2
MODEL_TYPE="gcn"
LEARNING_RATE=0.001
BATCH_SIZE=32
EPOCHS=100
DROPOUT=0.1
EARLY_STOPPING=10

# Run training script
echo "Starting model training at $(date)"

# Check if resuming from checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming training from checkpoint: $RESUME_CHECKPOINT"
    RESUME_FLAG="--resume_checkpoint $RESUME_CHECKPOINT"
else
    RESUME_FLAG=""
fi

srun --export=ALL python scripts/02_train_model.py \
  --preprocessed_dir ${PREPROCESSED_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --hidden_dim ${HIDDEN_DIM} \
  --num_layers ${NUM_LAYERS} \
  --model_type ${MODEL_TYPE} \
  --learning_rate ${LEARNING_RATE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --dropout ${DROPOUT} \
  --early_stopping_patience ${EARLY_STOPPING} \
  --checkpoint_dir ${EXPERIMENT_DIR}/checkpoints \
  --checkpoint_interval 10 \
  --device cuda \
  ${RESUME_FLAG}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
    
    # Save job information
    echo "Job ID: $SLURM_JOB_ID" > ${EXPERIMENT_DIR}/job_info.txt
    echo "Timestamp: $TIMESTAMP" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Preprocessed data: ${PREPROCESSED_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Output directory: ${OUTPUT_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Hyperparameters:" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Hidden dim: ${HIDDEN_DIM}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Num layers: ${NUM_LAYERS}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Model type: ${MODEL_TYPE}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Learning rate: ${LEARNING_RATE}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Batch size: ${BATCH_SIZE}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Epochs: ${EPOCHS}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Dropout: ${DROPOUT}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Early stopping: ${EARLY_STOPPING}" >> ${EXPERIMENT_DIR}/job_info.txt
    
    # Create symlink to latest experiment
    ln -sf ${EXPERIMENT_DIR} experiments/latest_train
    
    exit 0
else
    echo "Training failed with exit code $? at $(date)"
    exit 1
fi
