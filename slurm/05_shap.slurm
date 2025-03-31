#!/bin/bash
#SBATCH --job-name=GNN_SHAP
#SBATCH --output=logs/shap_%j.out
#SBATCH --error=logs/shap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=12:00:00
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
EXPERIMENT_DIR="experiments/shap_${TIMESTAMP}"
mkdir -p ${EXPERIMENT_DIR}/logs
mkdir -p ${EXPERIMENT_DIR}/visualizations

# Set environment variables
export OMP_NUM_THREADS=4

# Debugging output
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Set variables
PREPROCESSED_DIR="experiments/latest_preprocess/preprocessed"  # Use latest preprocessed data
MODEL_DIR="experiments/latest_train/model"  # Use latest trained model
OUTPUT_DIR="${EXPERIMENT_DIR}/results"

# SHAP analysis settings
BACKGROUND_SAMPLES=50
TEST_SAMPLES=20
TOP_N=10

# Run SHAP analysis script
echo "Starting SHAP analysis at $(date)"
srun --export=ALL python scripts/03_analyze_bottlenecks.py \
  --preprocessed_dir ${PREPROCESSED_DIR} \
  --model_dir ${MODEL_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --background_samples ${BACKGROUND_SAMPLES} \
  --test_samples ${TEST_SAMPLES} \
  --top_n ${TOP_N} \
  --device cuda \
  --visualization_dir ${EXPERIMENT_DIR}/visualizations

# Check exit status
if [ $? -eq 0 ]; then
    echo "SHAP analysis completed successfully at $(date)"
    
    # Save job information
    echo "Job ID: $SLURM_JOB_ID" > ${EXPERIMENT_DIR}/job_info.txt
    echo "Timestamp: $TIMESTAMP" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Preprocessed data: ${PREPROCESSED_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Model directory: ${MODEL_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "Output directory: ${OUTPUT_DIR}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "SHAP analysis settings:" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Background samples: ${BACKGROUND_SAMPLES}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Test samples: ${TEST_SAMPLES}" >> ${EXPERIMENT_DIR}/job_info.txt
    echo "  Top N: ${TOP_N}" >> ${EXPERIMENT_DIR}/job_info.txt
    
    # Create symlink to latest experiment
    ln -sf ${EXPERIMENT_DIR} experiments/latest_shap
    
    # Generate HTML report
    echo "Generating HTML report"
    python scripts/generate_report.py \
      --experiment_dir ${EXPERIMENT_DIR} \
      --preprocessed_dir ${PREPROCESSED_DIR} \
      --model_dir ${MODEL_DIR} \
      --shap_dir ${OUTPUT_DIR}
    
    exit 0
else
    echo "SHAP analysis failed with exit code $? at $(date)"
    exit 1
fi
