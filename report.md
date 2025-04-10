# 1M Jobs
Experiment 2:
    1,000,000 jobs
    Baseline GCN
    Test Loss: 0.9558
    Test RMSE: 0.9884

# Full Dataset 6M
Experiment 3:
    total jobs
    Baseline GCN
    Test Loss: 0.9483
    Test RMSE: 0.9738

    - Learning Rate: 0.001 - Epochs: 100 - Batch Size: 32 - Dropout: 0.1 - Early Stopping Patience: 10

# Based on Hyper Prameter Tunining Result
Experiment 4:
    total jobs
    Baseline GCN
    Test Loss: 0.6261
    Test RMSE: 0.7913 

    -hidden_dim 256 -num_layers 2 -learning_rate 0.0017331607338165434 -batch_size 64 -epochs 20 --dropout 0.23487409763750228
    
# Testing GAT
Experiment 5:
    total jobs
    Baseline GAT
    Test Loss: 
    Test RMSE: 

    -hidden_dim 256 -num_layers 2 -learning_rate 0.0017331607338165434 -batch_size 64 -epochs 20 --dropout 0.23487409763750228

# Testing Feature Selection

# MI not 0.32

# Testing Clustering

# Testing Ad + Clustering

# RayTune

# Gradiant Boosting