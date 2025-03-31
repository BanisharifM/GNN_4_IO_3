#!/bin/bash
# Master script to run the entire pipeline

# Set variables
DATA_DIR="/path/to/data"
PREPROCESSED_DIR="/path/to/preprocessed"
MODEL_DIR="/path/to/model"
HYPEROPT_DIR="/path/to/hyperopt_results"
SHAP_DIR="/path/to/shap_results"
SAMPLE_SIZE=1000  # Set to empty for full dataset or a number for testing

# Create directories
mkdir -p ${PREPROCESSED_DIR}
mkdir -p ${MODEL_DIR}
mkdir -p ${HYPEROPT_DIR}
mkdir -p ${SHAP_DIR}
mkdir -p logs

# Step 1: Preprocess data
echo "Step 1: Preprocessing data"
sbatch slurm/01_preprocess.slurm
# Wait for job to complete (in a real scenario, you would use job dependencies)
echo "Waiting for preprocessing to complete..."
sleep 10

# Step 2: Train model
echo "Step 2: Training model"
sbatch slurm/02_train.slurm
# Wait for job to complete
echo "Waiting for training to complete..."
sleep 10

# Step 3: Hyperparameter optimization (optional)
read -p "Do you want to run hyperparameter optimization? (y/n): " run_hyperopt
if [[ $run_hyperopt == "y" ]]; then
    echo "Step 3: Running hyperparameter optimization"
    sbatch slurm/03_hyperopt.slurm
    # Wait for job to complete
    echo "Waiting for hyperparameter optimization to complete..."
    sleep 10
fi

# Step 4: SHAP analysis
echo "Step 4: Running SHAP analysis"
sbatch slurm/04_shap.slurm
# Wait for job to complete
echo "Waiting for SHAP analysis to complete..."
sleep 10

echo "Pipeline execution submitted. Check job status with 'squeue -u $USER'"
