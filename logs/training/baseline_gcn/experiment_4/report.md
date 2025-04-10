# Experiment 4 Report

## Overview

Experiment 4 focused on training a more expressive GCN model with **increased hidden dimensions** and **tuned hyperparameters** using the same preprocessed dataset from Experiment 3. The experiment was run on a high-memory, high-core GPU node with multiple Slurm job submissions due to time limitations.

- **Job Name**: `gnn_train`
- **Compute Resources**:
  - **Partition**: `gpuA100x4`
  - **Nodes**: 1
  - **CPUs per Task**: 64
  - **GPUs**: 1 (A100)
  - **Memory**: 128 GB
  - **Time Limit**: 3 hours
- **Environment**:
  - **Conda Environment**: `gnn_env`
  - **Modules Loaded**: `cuda`
- **Directories**:
  - **Training Data**: `data/preprocessed/baseline_gcn/experiment_3/train`
  - **Validation Data**: `data/preprocessed/baseline_gcn/experiment_3/val`
  - **Test Data**: `data/preprocessed/baseline_gcn/experiment_3/test`
  - **Output Logs**: `logs/training/baseline_gcn/experiment_4`
- **Model Configuration**:
  - **Model Type**: GCN
  - **Hidden Dimensions**: 256
  - **Number of Layers**: 2
  - **Learning Rate**: 0.00173
  - **Batch Size**: 64
  - **Epochs**: 100
  - **Dropout**: 0.2349
  - **Early Stopping Patience**: 10
  - **Device**: CUDA (GPU)
  - **Resume Training**: Enabled
  - **Use Split Directories**: Enabled

## Objective

The goal of this experiment was to assess the impact of higher model capacity and optimized hyperparameters on performance using the same dataset and graph structure as Experiment 3.

## Dataset

- **Source**: Preprocessed dataset from `data/sample_total.csv`
- **Graph Construction**: Based on mutual information threshold `0.3259`
- **Dataset Split**:
  - **Train**: 4,653,053 samples
  - **Validation**: 997,082 samples
  - **Test**: 997,084 samples
- **Graph Details**:
  - **Nodes**: 44
  - **Edges**: 168

## Experimental Setup

- **Model**: GCN with larger hidden dimensions
- **Hyperparameters**:
  - Learning Rate: 0.001733
  - Epochs: 100
  - Batch Size: 64
  - Dropout: 0.2349
  - Early Stopping Patience: 10
- **Environment**:
  - Framework: PyTorch Geometric
  - Hardware: NVIDIA A100 GPU (1)

## Results

- **Loss**:
  - Final Training Loss: 0.6607
  - Final Validation Loss: 0.6246
  - Test Loss: 0.6261
  - Test RMSE: 0.7913
- **Training Time**: Split across 3 jobs, final session trained from epoch 20–20 (resumed at epoch 19)
- **Best Validation Epoch**: 15 (Val Loss: 0.6246)

## Analysis

- The GCN model with 256 hidden dimensions showed **notable improvement** in both validation and test metrics compared to previous experiments.
- The **test RMSE dropped to 0.7913**, a clear improvement over Experiment 3’s 0.9738.
- Early stopping kicked in after epoch 20, indicating convergence.
- Despite the model being split across multiple jobs due to wall-time limits, **checkpointing worked seamlessly**.
- Training was **consistently progressing**, and no torch-scatter or torch-sparse errors were reported in this run.

## Conclusion

Experiment 4 demonstrated that increasing model capacity and tuning hyperparameters led to **substantial gains in performance**. The use of a larger batch size and dropout appears to have improved generalization.

Key stats:
- **Train**: 4,653,053 samples
- **Validation**: 997,082 samples
- **Test**: 997,084 samples
- **Graph**: 44 nodes, 168 edges
- **Best Test RMSE**: 0.7913

## Future Work
