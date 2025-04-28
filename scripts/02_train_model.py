#!/usr/bin/env python
# Step 2: Train GNN model with clustering support

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pickle
import json
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from src.models.gnn import GNNRegressor

def create_dataset(node_features, targets, edge_index, edge_attr, cluster_labels=None):
    """
    Create a PyTorch Geometric dataset from node features, targets, and graph structure.
    
    Args:
        node_features: Tensor of node features [num_samples, num_nodes, feature_dim]
        targets: Tensor of target values [num_samples]
        edge_index: Tensor of edge indices [2, num_edges]
        edge_attr: Tensor of edge attributes [num_edges, edge_feature_dim]
        cluster_labels: Optional tensor of cluster labels [num_samples]
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    dataset = []
    
    for i in range(len(node_features)):
        data = Data(
            x=node_features[i].view(-1, 1),  # [num_nodes, 1]
            edge_index=edge_index,  # Same for all samples
            edge_attr=edge_attr,    # Same for all samples
            y=targets[i].view(-1)   # [1]
        )
        
        # Add cluster information if available
        if cluster_labels is not None:
            data.cluster = torch.tensor([cluster_labels[i]], dtype=torch.long)
        
        dataset.append(data)
    
    return dataset

def train_model(
    preprocessed_dir=None,
    train_dir=None,
    val_dir=None,
    test_dir=None,
    output_dir='logs/training',
    hidden_dim=64,
    num_layers=2,
    model_type='gcn',
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    dropout=0.1,
    early_stopping_patience=10,
    device=None,
    use_split_dirs=False,
    resume_training=True,
    use_clustering=False,
    cluster_as_feature=True,
    train_per_cluster=False
):
    """
    Train GNN model using preprocessed data.
    
    Args:
        preprocessed_dir (str): Directory with preprocessed data (for standard workflow)
        train_dir (str): Directory with training data (for split-based workflow)
        val_dir (str): Directory with validation data (for split-based workflow)
        test_dir (str): Directory with test data (for split-based workflow)
        output_dir (str): Directory to save model and results
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GNN layers
        model_type (str): Type of GNN ('gcn' or 'gat')
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        epochs (int): Number of epochs
        dropout (float): Dropout rate
        early_stopping_patience (int): Patience for early stopping
        device (str): Device to use ('cuda' or 'cpu')
        use_split_dirs (bool): Whether to use split-based workflow
        resume_training (bool): Whether to resume training from latest checkpoint if available
        use_clustering (bool): Whether to use clustering information
        cluster_as_feature (bool): Whether to use cluster as an additional feature
        train_per_cluster (bool): Whether to train separate models for each cluster
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Handle different data loading approaches based on workflow
    if use_split_dirs:
        # Split-based workflow: each split is in its own directory
        logger.info(f"Using split-based workflow with directories: {train_dir}, {val_dir}, {test_dir}")
        
        # Load edge structure (should be the same for all splits, so we use train_dir)
        edge_index = torch.load(os.path.join(train_dir, 'edge_index.pt'))
        edge_attr = torch.load(os.path.join(train_dir, 'edge_attr.pt'))
        
        # Load train data
        train_node_features = torch.load(os.path.join(train_dir, 'node_features.pt'))
        train_targets = torch.load(os.path.join(train_dir, 'targets.pt'))
        
        # Load val data
        val_node_features = torch.load(os.path.join(val_dir, 'node_features.pt'))
        val_targets = torch.load(os.path.join(val_dir, 'targets.pt'))
        
        # Load test data
        test_node_features = torch.load(os.path.join(test_dir, 'node_features.pt'))
        test_targets = torch.load(os.path.join(test_dir, 'targets.pt'))
        
        # Load cluster information if available and requested
        train_clusters = None
        val_clusters = None
        test_clusters = None
        
        if use_clustering:
            # Check if cluster data exists
            train_cluster_file = os.path.join(train_dir, 'data_with_clusters.csv')
            val_cluster_file = os.path.join(val_dir, 'data_with_clusters.csv')
            test_cluster_file = os.path.join(test_dir, 'data_with_clusters.csv')
            
            if os.path.exists(train_cluster_file) and os.path.exists(val_cluster_file) and os.path.exists(test_cluster_file):
                logger.info("Loading cluster information...")
                
                import pandas as pd
                train_df = pd.read_csv(train_cluster_file)
                val_df = pd.read_csv(val_cluster_file)
                test_df = pd.read_csv(test_cluster_file)
                
                train_clusters = torch.tensor(train_df['cluster'].values, dtype=torch.long)
                val_clusters = torch.tensor(val_df['cluster'].values, dtype=torch.long)
                test_clusters = torch.tensor(test_df['cluster'].values, dtype=torch.long)
                
                logger.info(f"Loaded cluster information: {len(train_clusters)} train, {len(val_clusters)} val, {len(test_clusters)} test")
            else:
                logger.warning("Clustering requested but cluster data not found. Proceeding without clustering.")
                use_clustering = False
        
        # Create datasets
        logger.info("Creating datasets from split directories...")
        train_dataset = create_dataset(train_node_features, train_targets, edge_index, edge_attr, train_clusters)
        val_dataset = create_dataset(val_node_features, val_targets, edge_index, edge_attr, val_clusters)
        test_dataset = create_dataset(test_node_features, test_targets, edge_index, edge_attr, test_clusters)
        
    else:
        # Original workflow: all data in one directory with indices
        logger.info(f"Loading preprocessed data from {preprocessed_dir}")
        
        edge_index = torch.load(os.path.join(preprocessed_dir, 'edge_index.pt'))
        edge_attr = torch.load(os.path.join(preprocessed_dir, 'edge_attr.pt'))
        node_features = torch.load(os.path.join(preprocessed_dir, 'node_features.pt'))
        targets = torch.load(os.path.join(preprocessed_dir, 'targets.pt'))
        
        # Load train/val/test splits
        train_indices = np.load(os.path.join(preprocessed_dir, 'train_indices.npy'))
        val_indices = np.load(os.path.join(preprocessed_dir, 'val_indices.npy'))
        test_indices = np.load(os.path.join(preprocessed_dir, 'test_indices.npy'))
        
        logger.info(f"Dataset size: {len(node_features)} samples")
        logger.info(f"Train/Val/Test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
        
        # Load cluster information if available and requested
        clusters = None
        
        if use_clustering:
            # Check if cluster data exists
            cluster_file = os.path.join(preprocessed_dir, 'data_with_clusters.csv')
            
            if os.path.exists(cluster_file):
                logger.info("Loading cluster information...")
                
                import pandas as pd
                df = pd.read_csv(cluster_file)
                clusters = torch.tensor(df['cluster'].values, dtype=torch.long)
                
                logger.info(f"Loaded cluster information for {len(clusters)} samples")
            else:
                logger.warning("Clustering requested but cluster data not found. Proceeding without clustering.")
                use_clustering = False
        
        # Create datasets
        logger.info("Creating datasets...")
        
        if clusters is not None:
            train_clusters = clusters[train_indices]
            val_clusters = clusters[val_indices]
            test_clusters = clusters[test_indices]
        else:
            train_clusters = None
            val_clusters = None
            test_clusters = None
        
        train_dataset = create_dataset(node_features[train_indices], targets[train_indices], edge_index, edge_attr, train_clusters)
        val_dataset = create_dataset(node_features[val_indices], targets[val_indices], edge_index, edge_attr, val_clusters)
        test_dataset = create_dataset(node_features[test_indices], targets[test_indices], edge_index, edge_attr, test_clusters)
    
    # If training per cluster is requested, handle that separately
    if use_clustering and train_per_cluster:
        return train_per_cluster_models(
            train_dataset, val_dataset, test_dataset,
            output_dir, hidden_dim, num_layers, model_type,
            learning_rate, batch_size, epochs, dropout,
            early_stopping_patience, device, resume_training
        )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    
    # Determine input dimension based on whether we're using cluster as feature
    input_dim = 1  # Default: single feature per node
    
    if use_clustering and cluster_as_feature:
        # Check if first item has cluster attribute
        if hasattr(train_dataset[0], 'cluster'):
            input_dim = 2  # Node feature + cluster feature
            logger.info(f"Using cluster as additional feature. Input dimension: {input_dim}")
        else:
            logger.warning("Cluster as feature requested but cluster data not found. Using default input dimension.")
    
    # Create model
    logger.info(f"Creating {model_type.upper()} model with {num_layers} layers and {hidden_dim} hidden dimensions")
    
    model = GNNRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,  # Single target value
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Variables for training
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Define checkpoint paths
    latest_checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pt')
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    
    # Check for existing checkpoints to resume training
    if resume_training and os.path.exists(latest_checkpoint_path):
        logger.info(f"Found checkpoint at {latest_checkpoint_path}. Resuming training...")
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get training state
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        
        logger.info(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.6f} at epoch {best_epoch+1}")
    else:
        logger.info("No checkpoint found or resume disabled. Starting training from scratch.")
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs (from epoch {start_epoch+1})")
    
    start_time = time.time() if start_epoch == 0 else time.time() - sum(train_losses) * 0  # Placeholder for tracking time
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Add cluster as feature if requested
            if use_clustering and cluster_as_feature and hasattr(batch, 'cluster'):
                # Normalize cluster to [0,1] range for numerical stability
                num_clusters = torch.max(batch.cluster).item() + 1
                normalized_cluster = batch.cluster.float() / num_clusters
                
                # Reshape to match node feature dimensions
                cluster_feature = normalized_cluster.view(-1, 1).repeat_interleave(batch.x.size(0) // batch.num_graphs, dim=0)
                
                # Concatenate with node features
                batch.x = torch.cat([batch.x, cluster_feature], dim=1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out = out.squeeze(-1)
            
            # Compute loss
            loss = criterion(out, batch.y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch.num_graphs
        
        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Add cluster as feature if requested
                if use_clustering and cluster_as_feature and hasattr(batch, 'cluster'):
                    # Normalize cluster to [0,1] range for numerical stability
                    num_clusters = torch.max(batch.cluster).item() + 1
                    normalized_cluster = batch.cluster.float() / num_clusters
                    
                    # Reshape to match node feature dimensions
                    cluster_feature = normalized_cluster.view(-1, 1).repeat_interleave(batch.x.size(0) // batch.num_graphs, dim=0)
                    
                    # Concatenate with node features
                    batch.x = torch.cat([batch.x, cluster_feature], dim=1)
                
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                out = out.squeeze(-1)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save latest checkpoint (single file that gets overwritten)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model_config': {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'model_type': model_type,
                'use_clustering': use_clustering,
                'cluster_as_feature': cluster_as_feature
            }
        }
        
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Calculate training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    # Load best model for evaluation
    logger.info(f"Loading best model from epoch {best_epoch+1} for evaluation")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Add cluster as feature if requested
            if use_clustering and cluster_as_feature and hasattr(batch, 'cluster'):
                # Normalize cluster to [0,1] range for numerical stability
                num_clusters = torch.max(batch.cluster).item() + 1
                normalized_cluster = batch.cluster.float() / num_clusters
                
                # Reshape to match node feature dimensions
                cluster_feature = normalized_cluster.view(-1, 1).repeat_interleave(batch.x.size(0) // batch.num_graphs, dim=0)
                
                # Concatenate with node features
                batch.x = torch.cat([batch.x, cluster_feature], dim=1)
            
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out = out.squeeze(-1)
            loss = criterion(out, batch.y)
            test_loss += loss.item() * batch.num_graphs
            
            # Store predictions and targets for analysis
            predictions.extend(out.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    logger.info(f"Test Loss: {test_loss:.6f}")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test MAE: {mae:.6f}")
    
    # Save metrics
    metrics = {
        'test_loss': test_loss,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1,
        'total_epochs': epoch + 1,
        'training_time': total_time
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    np.savez(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=predictions,
        targets=targets
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 5))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'))
    
    logger.info(f"Results saved to {output_dir}")
    
    return metrics

def train_per_cluster_models(
    train_dataset, val_dataset, test_dataset,
    output_dir, hidden_dim, num_layers, model_type,
    learning_rate, batch_size, epochs, dropout,
    early_stopping_patience, device, resume_training
):
    """
    Train separate models for each cluster.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        output_dir: Directory to save models and results
        hidden_dim: Hidden dimension size
        num_layers: Number of GNN layers
        model_type: Type of GNN ('gcn' or 'gat')
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        dropout: Dropout rate
        early_stopping_patience: Patience for early stopping
        device: Device to use ('cuda' or 'cpu')
        resume_training: Whether to resume training from latest checkpoint if available
        
    Returns:
        Dictionary of metrics for each cluster
    """
    logger.info("Training separate models for each cluster")
    
    # Group datasets by cluster
    train_by_cluster = {}
    val_by_cluster = {}
    test_by_cluster = {}
    
    # Group training data
    for data in train_dataset:
        if hasattr(data, 'cluster'):
            cluster_id = data.cluster.item()
            if cluster_id not in train_by_cluster:
                train_by_cluster[cluster_id] = []
            train_by_cluster[cluster_id].append(data)
    
    # Group validation data
    for data in val_dataset:
        if hasattr(data, 'cluster'):
            cluster_id = data.cluster.item()
            if cluster_id not in val_by_cluster:
                val_by_cluster[cluster_id] = []
            val_by_cluster[cluster_id].append(data)
    
    # Group test data
    for data in test_dataset:
        if hasattr(data, 'cluster'):
            cluster_id = data.cluster.item()
            if cluster_id not in test_by_cluster:
                test_by_cluster[cluster_id] = []
            test_by_cluster[cluster_id].append(data)
    
    # Get all unique cluster IDs
    all_clusters = set(train_by_cluster.keys()) | set(val_by_cluster.keys()) | set(test_by_cluster.keys())
    logger.info(f"Found {len(all_clusters)} clusters: {sorted(all_clusters)}")
    
    # Train a model for each cluster
    cluster_metrics = {}
    
    for cluster_id in sorted(all_clusters):
        logger.info(f"Training model for cluster {cluster_id}")
        
        # Create cluster-specific output directory
        cluster_output_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_output_dir, exist_ok=True)
        
        # Skip if no training data for this cluster
        if cluster_id not in train_by_cluster or len(train_by_cluster[cluster_id]) == 0:
            logger.warning(f"No training data for cluster {cluster_id}. Skipping.")
            continue
        
        # Skip if no validation data for this cluster
        if cluster_id not in val_by_cluster or len(val_by_cluster[cluster_id]) == 0:
            logger.warning(f"No validation data for cluster {cluster_id}. Skipping.")
            continue
        
        # Create data loaders
        train_loader = DataLoader(train_by_cluster[cluster_id], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_by_cluster.get(cluster_id, []), batch_size=batch_size)
        
        # Create test loader if test data exists for this cluster
        test_loader = None
        if cluster_id in test_by_cluster and len(test_by_cluster[cluster_id]) > 0:
            test_loader = DataLoader(test_by_cluster[cluster_id], batch_size=batch_size)
        
        logger.info(f"Cluster {cluster_id} data sizes - Train: {len(train_by_cluster[cluster_id])}, "
                   f"Val: {len(val_by_cluster.get(cluster_id, []))}, "
                   f"Test: {len(test_by_cluster.get(cluster_id, []))}")
        
        # Create model
        model = GNNRegressor(
            input_dim=1,  # Single feature per node (no need to use cluster as feature here)
            hidden_dim=hidden_dim,
            output_dim=1,  # Single target value
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        ).to(device)
        
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Variables for training
        start_epoch = 0
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Define checkpoint paths
        latest_checkpoint_path = os.path.join(cluster_output_dir, 'latest_checkpoint.pt')
        best_model_path = os.path.join(cluster_output_dir, 'best_model.pt')
        
        # Check for existing checkpoints to resume training
        if resume_training and os.path.exists(latest_checkpoint_path):
            logger.info(f"Found checkpoint for cluster {cluster_id}. Resuming training...")
            
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Get training state
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', 0)
        
        # Training loop
        logger.info(f"Starting training for cluster {cluster_id} for {epochs} epochs (from epoch {start_epoch+1})")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = batch.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                out = out.squeeze(-1)
                
                # Compute loss
                loss = criterion(out, batch.y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch.num_graphs
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    out = out.squeeze(-1)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item() * batch.num_graphs
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Log progress
            logger.info(f"Cluster {cluster_id}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save latest checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'model_config': {
                    'input_dim': 1,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'model_type': model_type,
                    'cluster_id': cluster_id
                }
            }
            
            torch.save(checkpoint, latest_checkpoint_path)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save(checkpoint, best_model_path)
                logger.info(f"New best model for cluster {cluster_id} saved at epoch {epoch+1} with validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered for cluster {cluster_id} after {epoch+1} epochs")
                    break
        
        # Calculate training time
        total_time = time.time() - start_time
        
        # Load best model for evaluation
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate on test set if available
        test_metrics = {}
        
        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    out = out.squeeze(-1)
                    loss = criterion(out, batch.y)
                    test_loss += loss.item() * batch.num_graphs
                    
                    # Store predictions and targets for analysis
                    predictions.extend(out.cpu().numpy())
                    targets.extend(batch.y.cpu().numpy())
            
            test_loss = test_loss / len(test_loader.dataset)
            
            # Convert to numpy arrays
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            
            logger.info(f"Cluster {cluster_id} - Test Loss: {test_loss:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
            
            # Save metrics
            test_metrics = {
                'test_loss': test_loss,
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mae': mae
            }
            
            # Save predictions
            np.savez(
                os.path.join(cluster_output_dir, 'predictions.npz'),
                predictions=predictions,
                targets=targets
            )
            
            # Plot predictions vs targets
            plt.figure(figsize=(10, 5))
            plt.scatter(targets, predictions, alpha=0.5)
            plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Cluster {cluster_id} - Predictions vs True Values')
            plt.grid(True)
            plt.savefig(os.path.join(cluster_output_dir, 'predictions_vs_targets.png'))
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Cluster {cluster_id} - Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(cluster_output_dir, 'loss_curve.png'))
        
        # Save all metrics
        metrics = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch + 1,
            'total_epochs': epoch + 1,
            'training_time': total_time,
            'train_size': len(train_by_cluster[cluster_id]),
            'val_size': len(val_by_cluster.get(cluster_id, [])),
            'test_size': len(test_by_cluster.get(cluster_id, [])),
            **test_metrics
        }
        
        # Convert all numpy types to native Python types
        metrics = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
 
        with open(os.path.join(cluster_output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Store metrics for this cluster
        cluster_metrics[cluster_id] = metrics
    
    # Save overall metrics
    overall_metrics = {
        'num_clusters': len(all_clusters),
        'clusters': sorted(list(all_clusters)),
        'cluster_metrics': cluster_metrics
    }
    
    with open(os.path.join(output_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    logger.info(f"Cluster-based training completed. Results saved to {output_dir}")
    
    return overall_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GNN model')
    
    # Data arguments
    parser.add_argument('--preprocessed_dir', type=str, help='Directory with preprocessed data (for standard workflow)')
    parser.add_argument('--train_dir', type=str, help='Directory with training data (for split-based workflow)')
    parser.add_argument('--val_dir', type=str, help='Directory with validation data (for split-based workflow)')
    parser.add_argument('--test_dir', type=str, help='Directory with test data (for split-based workflow)')
    parser.add_argument('--output_dir', type=str, default='logs/training', help='Directory to save model and results')
    parser.add_argument('--use_split_dirs', action='store_true', help='Whether to use split-based workflow')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat'], help='Type of GNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--no_resume', action='store_true', help='Disable resuming from checkpoint')
    
    # Clustering arguments
    parser.add_argument('--use_clustering', action='store_true', help='Whether to use clustering information')
    parser.add_argument('--cluster_as_feature', action='store_true', help='Whether to use cluster as an additional feature')
    parser.add_argument('--train_per_cluster', action='store_true', help='Whether to train separate models for each cluster')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Validate arguments
    if args.use_split_dirs and (args.train_dir is None or args.val_dir is None or args.test_dir is None):
        raise ValueError("When using split directories, train_dir, val_dir, and test_dir must be provided")
    
    if not args.use_split_dirs and args.preprocessed_dir is None:
        raise ValueError("When not using split directories, preprocessed_dir must be provided")
    
    if args.train_per_cluster and not args.use_clustering:
        raise ValueError("train_per_cluster requires use_clustering to be enabled")
    
    # Train model
    train_model(
        preprocessed_dir=args.preprocessed_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        use_split_dirs=args.use_split_dirs,
        resume_training=not args.no_resume,
        use_clustering=args.use_clustering,
        cluster_as_feature=args.cluster_as_feature,
        train_per_cluster=args.train_per_cluster
    )

if __name__ == "__main__":
    main()
