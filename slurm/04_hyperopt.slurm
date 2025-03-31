#!/bin/bash
#SBATCH --job-name=GNN_Hyperopt
#SBATCH --output=logs/hyperopt_%j.out
#SBATCH --error=logs/hyperopt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=48:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:2
#SBATCH --account=bdau-delta-gpu

# Load necessary modules
module load python/3.11.6
module load cuda/11.8

# Activate virtual environment (replace with your environment path)
source /path/to/your/venv/bin/activate

# Create experiment directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="experiments/hyperopt_${TIMESTAMP}"
mkdir -p ${EXPERIMENT_DIR}/logs
mkdir -p ${EXPERIMENT_DIR}/ray_results

# Set environment variables
export OMP_NUM_THREADS=4

# Ray-specific environment variables
export RAY_ADDRESS="auto"
export RAY_OBJECT_STORE_MEMORY=50000000000  # 50GB

# Debugging output
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Set variables
PREPROCESSED_DIR="experiments/latest_preprocess/preprocessed"  # Use latest preprocessed data
OUTPUT_DIR="${EXPERIMENT_DIR}/results"

# Hyperparameter optimization settings
NUM_SAMPLES=20
MAX_EPOCHS=100
EARLY_STOPPING=10
CPUS_PER_TRIAL=4
GPUS_PER_TRIAL=0.5
NUM_WORKERS=4

# Run hyperparameter optimization script
echo "Starting hyperparameter optimization at $(date)"
srun --export=ALL python scripts/04_hyperparameter_optimization.py \
  --preprocessed_dir ${PREPROCESSED_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --num_samples ${NUM_SAMPLES} \
  --max_epochs ${MAX_EPOCHS} \
  --early_stopping_patience ${EARLY_STOPPING} \
  --cpus_per_trial ${CPUS_PER_TRIAL} \
  --gpus_per_trial ${GPUS_PER_TRIAL} \
  --num_workers ${NUM_WORKERS} \
  --ray_dir ${EXPERIMENT_DIR}/ray_results

# Check exit status
if [ $? -eq 0 ]; then
    echo "Hyperparameter optimization completed successfully at $(date)"
    
    # Save job information
    echo "Job ID: $SLURM_JOB_ID" > ${EXPERIMENT_DIR}/job_info.txt
    echo "Timestamp: $TIMESTAMP" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Preprocessed data: ${PREPROCESSED_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Output directory: ${OUTPUT_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Hyperparameter optimization settings:" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Num samples: ${NUM_SAMPLES}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Max epochs: ${MAX_EPOCHS}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Early stopping: ${EARLY_STOPPING}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  CPUs per trial: ${CPUS_PER_TRIAL}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  GPUs per trial: ${GPUS_PER_TRIAL}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Num workers: ${NUM_WORKERS}" >> ${EXPERIMENT_DIR}/job_info.txt
    
    # Create symlink to latest experiment
    ln -sf ${EXPERIMENT_DIR} experiments/latest_hyperopt
    
    # Submit job to train with best hyperparameters
    echo "Submitting job to train with best hyperparameters"
    chmod +x ${OUTPUT_DIR}/train_best_model.sh
    sbatch ${OUTPUT_DIR}/train_best_model.sh
    
    exit 0
else
    echo "Hyperparameter optimization failed with exit code $? at $(date)"
    exit 1
fi
