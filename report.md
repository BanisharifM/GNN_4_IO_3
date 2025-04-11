# Test Pipeline for 100 Jobs
Experiment 1

# Run GCN for 1M Jobs
Experiment 2:
    1,000,000 jobs
    Baseline GCN
    Test Loss: 0.9558
    Test RMSE: 0.9884

# Run GCN for total dataset(6M Jobs)
Experiment 3:
    total > 6M jobs
    Baseline GCN
    Test Loss: 0.9483
    Test RMSE: 0.9738

    - Learning Rate: 0.001 - Epochs: 100 - Batch Size: 32 - Dropout: 0.1 - Early Stopping Patience: 10

# Run GCN Based on Hyper Prameter Tunining Result
Experiment 4:
    total > 6M jobs
    Baseline GCN
    Test Loss: 0.6261
    Test RMSE: 0.7913 

    -hidden_dim 256 -num_layers 2 -learning_rate 0.0017331607338165434 -batch_size 64 -epochs 20 --dropout 0.23487409763750228
    
# Run GAT for total dataset(6M Jobs) + Exp4 Hyper Prameters
Experiment 5:
    total > 6M jobs
    Baseline GAT
    Test Loss: 0.7588
    Test RMSE: 0.8711 

    -hidden_dim 256 -num_layers 2 -learning_rate 0.0017331607338165434 -batch_size 64 -epochs 20 --dropout 0.23487409763750228

# RayTune for 40 Sample
Experiment 6:


# Testing Feature Selection

# MI not 0.32

# Testing Clustering

# Testing Ad + Clustering

# RayTune

# Gradiant Boosting