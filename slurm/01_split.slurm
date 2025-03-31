#!/bin/bash
#SBATCH --job-name=GNN_Split
#SBATCH --output=logs/split_%j.out
#SBATCH --error=logs/split_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --account=bdau-delta-gpu

# Load necessary modules
module load python/3.11.6
module load cuda/11.8

# Activate virtual environment (replace with your environment path)
source /path/to/your/venv/bin/activate

# Create experiment directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="experiments/split_${TIMESTAMP}"
mkdir -p ${EXPERIMENT_DIR}/logs

# Set environment variables
export OMP_NUM_THREADS=4

# Debugging output
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Set variables
DATA_DIR="/path/to/data"
OUTPUT_DIR="${EXPERIMENT_DIR}/split_data"
SAMPLE_SIZE=  # Leave empty for full dataset or set a number for testing

# Run data splitting script
echo "Starting data splitting at $(date)"
srun --export=ALL python scripts/00_split_data.py \
  --data_file ${DATA_DIR}/sample_train_100.csv \
  --output_dir ${OUTPUT_DIR} \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --random_seed 42 \
  --sample_size ${SAMPLE_SIZE}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Data splitting completed successfully at $(date)"
    
    # Save job information
    echo "Job ID: $SLURM_JOB_ID" > ${EXPERIMENT_DIR}/job_info.txt
    echo "Timestamp: $TIMESTAMP" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Data file: ${DATA_DIR}/sample_train_100.csv" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Output directory: ${OUTPUT_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    
    # Create symlink to latest experiment
    ln -sf ${EXPERIMENT_DIR} experiments/latest_split
    
    exit 0
else
    echo "Data splitting failed with exit code $? at $(date)"
    exit 1
fi
