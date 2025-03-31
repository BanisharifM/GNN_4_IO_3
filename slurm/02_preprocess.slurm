#!/bin/bash
#SBATCH --job-name=GNN_Preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=06:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --account=bdau-delta-gpu

# Load necessary modules
module load python/3.11.6
module load cuda/11.8

# Activate virtual environment (replace with your environment path)
source /path/to/your/venv/bin/activate

# Create experiment directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="experiments/preprocess_${TIMESTAMP}"
mkdir -p ${EXPERIMENT_DIR}/logs

# Set environment variables
export OMP_NUM_THREADS=4

# Debugging output
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Set variables
SPLIT_DIR="experiments/latest_split/split_data"  # Use latest split data
MI_FILE="/path/to/data/mutual_information2.csv"
OUTPUT_DIR="${EXPERIMENT_DIR}/preprocessed"
MI_THRESHOLD=0.3259

# Run preprocessing script
echo "Starting preprocessing at $(date)"
srun --export=ALL python scripts/01_preprocess_data.py \
  --data_file ${SPLIT_DIR}/train.csv \
  --mi_file ${MI_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --mi_threshold ${MI_THRESHOLD}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully at $(date)"
    
    # Save job information
    echo "Job ID: $SLURM_JOB_ID" > ${EXPERIMENT_DIR}/job_info.txt
    echo "Timestamp: $TIMESTAMP" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Data file: ${SPLIT_DIR}/train.csv" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "MI file: ${MI_FILE}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "MI threshold: ${MI_THRESHOLD}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Output directory: ${OUTPUT_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    
    # Create symlink to latest experiment
    ln -sf ${EXPERIMENT_DIR} experiments/latest_preprocess
    
    exit 0
else
    echo "Preprocessing failed with exit code $? at $(date)"
    exit 1
fi
