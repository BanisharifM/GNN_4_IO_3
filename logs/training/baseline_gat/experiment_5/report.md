# Experiment 5 Report

## Overview

This experiment introduced a new model architecture — the **Graph Attention Network (GAT)** — to evaluate its performance compared to the baseline GCN models. The training was conducted using Slurm with checkpointing across multiple jobs due to time constraints.

- **Job Name**: `gnn_train`
- **Compute Resources**:
  - **Partition**: `gpuA100x4-interactive`
  - **Nodes**: 1
  - **CPUs per Task**: 64
  - **GPUs**: 1 (A100)
  - **Memory**: 128 GB
  - **Time Limit per Job**: 1 hour (split across 3 jobs)
- **Environment**:
  - **Conda Environment**: `gnn_env`
  - **Modules Loaded**: `cuda`
- **Directories**:
  - **Training Data**: `data/preprocessed/baseline_gcn/experiment_3/train`
  - **Validation Data**: `data/preprocessed/baseline_gcn/experiment_3/val`
  - **Test Data**: `data/preprocessed/baseline_gcn/experiment_3/test`
  - **Output Logs**: `logs/training/baseline_gat/experiment_5`
- **Model Configuration**:
  - **Model Type**: GAT (Graph Attention Network)
  - **Hidden Dimensions**: 256
  - **Number of Layers**: 2
  - **Learning Rate**: 0.0017331607338165434
  - **Batch Size**: 64
  - **Epochs**: 100
  - **Dropout**: 0.23487409763750228
  - **Early Stopping Patience**: 10
  - **Device**: CUDA (GPU)
  - **Resume Training**: Enabled
  - **Use Split Directories**: Enabled

## Objective

The aim was to evaluate whether incorporating attention mechanisms via GAT improves model performance over GCN in the same experimental setup using mutual information-based graph structures.

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

- **Model**: GAT with attention-based node aggregation
- **Hyperparameters**:
  - Hidden Dimensions: 256
  - Number of Layers: 2
  - Learning Rate: 0.0017331607338165434
  - Batch Size: 64
  - Dropout: 0.23487409763750228
  - Epochs: 100
  - Early Stopping Patience: 10
- **Environment**:
  - Framework: PyTorch Geometric
  - Hardware: NVIDIA A100 GPU (1)

## Results

- **Loss**:
  - Final Training Loss: 0.9496
  - Final Validation Loss: 0.9490
  - Test Loss: **0.7588**
  - Test RMSE: **0.8711**
- **Training Time**: ~3 hours across 3 resumed jobs
- **Best Validation Epoch**: Epoch 1 (Val Loss: 0.7590)

## Analysis

- GAT demonstrated a **better test loss** (0.7588) compared to GCN in Experiment 3 (0.9483) and Experiment 4 (0.6261).
- Despite early convergence of validation loss and training loss plateauing, GAT produced a **significantly improved RMSE** compared to GCN in Experiment 3 (0.9738), though still slightly worse than GCN in Experiment 4 (0.7913).
- The model may have overfitted after the first few epochs as validation loss did not improve, suggesting **attention-based models may benefit from stricter regularization or learning rate decay.**
- The training was consistently resumed from checkpoints without issues.

## Conclusion

GAT achieved a **strong test RMSE of 0.8711**, validating that attention mechanisms can improve generalization on this dataset. However, further tuning or deeper architectures might be required to surpass the best-performing GCN from Experiment 4.

Key stats:
- **Train**: 4,653,053 samples
- **Validation**: 997,082 samples
- **Test**: 997,084 samples
- **Graph Structure**: 44 nodes, 168 edges
- **Best Test RMSE**: 0.8711

## Future Work

- Explore multi-head attention in GAT.
- Consider using GATv2 or Graph Transformer architectures.
- Apply learning rate scheduling and stronger regularization.
- Use attention visualization tools to interpret the learned focus of the GAT model.
