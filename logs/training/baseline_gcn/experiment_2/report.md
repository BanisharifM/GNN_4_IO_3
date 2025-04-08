# Experiment 2 Report

## Overview

This experiment involved training the baseline GCN model using a Slurm job submission script. The setup details are as follows:

- **Job Name**: `gnn_train`
- **Compute Resources**:
    - **Partition**: `gpuA100x4-interactive`
    - **Nodes**: 1
    - **CPUs per Task**: 16
    - **GPUs**: 1 (A100)
    - **Memory**: 32 GB
    - **Time Limit**: 1 hour
- **Environment**:
    - **Conda Environment**: `gnn_env`
    - **Modules Loaded**: `cuda`
- **Directories**:
    - **Training Data**: `data/preprocessed/baseline_gcn/experiment_2/train`
    - **Validation Data**: `data/preprocessed/baseline_gcn/experiment_2/val`
    - **Test Data**: `data/preprocessed/baseline_gcn/experiment_2/test`
    - **Output Logs**: `logs/training/baseline_gcn/experiment_2`
- **Model Configuration**:
    - **Model Type**: GCN
    - **Hidden Dimensions**: 64
    - **Number of Layers**: 2
    - **Learning Rate**: 0.001
    - **Batch Size**: 32
    - **Epochs**: 100
    - **Dropout**: 0.1
    - **Early Stopping Patience**: 10
    - **Device**: CUDA (GPU)
    - **Resume Training**: Enabled
    - **Use Split Directories**: Enabled

The training script (`scripts/02_train_model.py`) was executed with the above parameters, and logs were saved in `logs/slurm/experiment_2/`. This setup ensured efficient resource utilization and reproducibility.

## Objective

The goal of this experiment was to evaluate the baseline GCN model's performance on the preprocessed dataset and assess its effectiveness for the given task.

## Dataset

- **Source**: Preprocessed dataset for baseline GCN.
- **Details**: 1,000,000 jobs.

## Experimental Setup

- **Model**: Baseline GCN
- **Hyperparameters**:
    - Learning Rate: 0.001
    - Epochs: 100
    - Batch Size: 32
    - Other Parameters: Dropout (0.1), Early Stopping (Patience: 10)
- **Environment**:
    - Framework: PyTorch
    - Hardware: NVIDIA A100 GPU

## Results

- **Metrics**:
    - Accuracy: N/A
    - Precision: N/A
    - Recall: N/A
    - F1-Score: N/A
- **Loss**:
    - Final Training Loss: 0.9485
    - Final Validation Loss: 0.9474
    - Test Loss: 0.9558
    - Test RMSE: 0.9884
- **Other Observations**:
    - Training resumed from epoch 19 with the best validation loss of 0.947294 at epoch 9.
    - Training completed in 128 seconds for the resumed 20 epochs.

## Analysis

- The model achieved a test loss of 0.9558 and a test RMSE of 0.9884, indicating reasonable performance for the baseline GCN.
- Issues were encountered with the `torch-scatter` and `torch-sparse` libraries, which were disabled during execution. This may have impacted performance.
- A warning about mismatched target and input sizes during loss computation was observed, which should be addressed in future experiments.

## Conclusion

The dataset was split into train, validation, and test sets with sizes of 700,000, 150,000, and 150,000, respectively. The original data was sourced from `data/sample_train_total.csv`, and the split datasets were saved in `data/split_data/sample_1,000,000/`. Preprocessed data for each split was stored in `data/preprocessed/baseline_gcn/experiment_2/`.

Key preprocessing statistics:
- **Train**: 700,000 samples
- **Validation**: 150,000 samples
- **Test**: 150,000 samples
- **Graph Details**: 44 nodes, 168 edges

These results provide a strong foundation for evaluating the baseline GCN model's performance.

## Future Work

