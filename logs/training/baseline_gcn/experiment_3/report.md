# Experiment 3 Report

## Overview

This experiment involved training the baseline GCN model using Slurm job submission on preprocessed data with mutual information (MI)-based graph construction. The experiment leveraged Slurm scheduling and a GPU-accelerated environment to conduct the training.

- **Job Name**: `gnn_train`
- **Compute Resources**:
  - **Partition**: `gpuA100x4-interactive`
  - **Nodes**: 1
  - **CPUs per Task**: 16
  - **GPUs**: 1 (A100)
  - **Memory**: 32 GB
  - **Time Limit**: 1 hour
- **Environment**:
  - **Virtual Environment**: `gnn-env` (created with `venv`)
  - **Modules Loaded**: `cuda`
- **Directories**:
  - **Training Data**: `data/preprocessed/baseline_gcn/experiment_3/train`
  - **Validation Data**: `data/preprocessed/baseline_gcn/experiment_3/val`
  - **Test Data**: `data/preprocessed/baseline_gcn/experiment_3/test`
  - **Output Logs**: `logs/training/baseline_gcn/experiment_3`
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

The training script (`scripts/02_train_model.py`) was executed, and the logs were recorded under `logs/slurm/experiment_3`.

## Objective

The goal of this experiment was to assess the GCN model’s performance using MI-based graph structures and to continue evaluation from the baseline set up in Experiment 2.

## Dataset

- **Source**: Preprocessed dataset from `data/sample_total.csv`
- **Preprocessing**: Based on mutual information threshold (0.3259)
- **Dataset Split**:
  - **Train**: 4,653,053 samples
  - **Validation**: 997,082 samples
  - **Test**: 997,084 samples
- **Graph Details**:
  - **Nodes**: 44
  - **Edges**: 168

## Experimental Setup

- **Model**: Baseline GCN
- **Hyperparameters**:
  - Learning Rate: 0.001
  - Epochs: 100
  - Batch Size: 32
  - Dropout: 0.1
  - Early Stopping Patience: 10
- **Environment**:
  - Framework: PyTorch Geometric
  - Hardware: NVIDIA A100 GPU

## Results

- **Loss**:
  - Final Training Loss: 0.9495 (plateaued early)
  - Final Validation Loss: 0.9489
  - Test Loss: 0.9483
  - Test RMSE: 0.9738
- **Training Time**: ~4.5 hours (across 2 job submissions)
- **Epochs**:
  - Initial job was killed at epoch 4 due to time limit.
  - Training resumed and completed with early stopping at epoch 24.
  - Best model saved at epoch 14.

## Analysis

- The model successfully resumed from the last saved checkpoint and converged to a stable validation loss of 0.9489.
- **Torch-Scatter** and **Torch-Sparse** failed to load due to mismatched binary compatibility, which may have negatively affected performance by falling back to less efficient operations.
- The training loss plateaued very early, suggesting potential underfitting or lack of model complexity.
- Despite this, the test RMSE of 0.9738 indicates slightly better generalization compared to Experiment 2.

## Conclusion

Experiment 3 improved preprocessing by leveraging mutual information to construct more meaningful graphs, although performance gains remained marginal compared to Experiment 2.

Key preprocessing statistics:
- **Train**: 4,653,053 samples
- **Validation**: 997,082 samples
- **Test**: 997,084 samples
- **Graph Structure**: 44 nodes, 168 edges

These results serve as a solid foundation for further experiments involving model architecture tuning or alternative graph construction strategies.

## Future Work
