#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GNN model for I/O bottleneck prediction
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.models import GNNRegressor
from src.models.gnn import GNNRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_dataset(node_features, targets, edge_index, edge_attr):
    """Create a PyTorch Geometric dataset from node features and targets."""
    dataset = []
    for i in range(len(node_features)):
        data = Data(
            x=node_features[i].view(-1, 1),  # [num_nodes, 1]
            edge_index=edge_index,           # Same for all samples
            edge_attr=edge_attr,             # Same for all samples
            y=targets[i].view(-1)            # [1]
        )
        dataset.append(data)
    return dataset

def train_model(
    preprocessed_dir=None,
    train_dir=None,
    val_dir=None,
    test_dir=None,
    output_dir='logs/training',
    model_type='gcn',
    hidden_dim=64,
    num_layers=2,
    dropout=0.1,
    learning_rate=0.001,
    weight_decay=5e-4,
    batch_size=32,
    epochs=100,
    patience=20,
    device=None,
    use_split_dirs=False,
):
    """Train a GNN model for I/O bottleneck prediction."""
    
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
        train_data = torch.load(os.path.join(train_dir, 'train_data.pt'))
        if not train_data:  # If train_data.pt doesn't exist, try to create it from components
            node_features = torch.load(os.path.join(train_dir, 'node_features.pt'))
            targets = torch.load(os.path.join(train_dir, 'targets.pt'))
            train_data = create_dataset(node_features, targets, edge_index, edge_attr)
        
        # Load val data
        val_data = torch.load(os.path.join(val_dir, 'val_data.pt'))
        if not val_data:  # If val_data.pt doesn't exist, try to create it from components
            node_features = torch.load(os.path.join(val_dir, 'node_features.pt'))
            targets = torch.load(os.path.join(val_dir, 'targets.pt'))
            val_data = create_dataset(node_features, targets, edge_index, edge_attr)
        
        # Load test data
        test_data = torch.load(os.path.join(test_dir, 'test_data.pt'))
        if not test_data:  # If test_data.pt doesn't exist, try to create it from components
            node_features = torch.load(os.path.join(test_dir, 'node_features.pt'))
            targets = torch.load(os.path.join(test_dir, 'targets.pt'))
            test_data = create_dataset(node_features, targets, edge_index, edge_attr)
        
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
        
        # Create datasets
        logger.info("Creating datasets...")
        
        train_data = create_dataset(node_features[train_indices], targets[train_indices], edge_index, edge_attr)
        val_data = create_dataset(node_features[val_indices], targets[val_indices], edge_index, edge_attr)
        test_data = create_dataset(node_features[test_indices], targets[test_indices], edge_index, edge_attr)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    logger.info(f"Train/Val/Test sizes: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    
    # Create model
    logger.info(f"Creating {model_type.upper()} model with {num_layers} layers and {hidden_dim} hidden dimensions")
    
    model = GNNRegressor(
        input_dim=1,  # Single feature per node
        hidden_dim=hidden_dim,
        output_dim=1,  # Single target value
        num_layers=num_layers,
        dropout=dropout,
        gnn_type=model_type
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs with patience {patience}")
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_data)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
        
        val_loss /= len(val_data)
        val_losses.append(val_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            logger.info(f"Saved best model at epoch {epoch+1} with validation loss {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} after {patience} epochs without improvement")
                break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            test_loss += loss.item() * batch.num_graphs
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())
    
    test_loss /= len(test_data)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
    
    # Save results
    results = {
        'model_type': model_type,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {os.path.join(output_dir, 'results.json')}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model for I/O bottleneck prediction')
    
    # Data parameters
    parser.add_argument('--preprocessed_dir', type=str, default=None, help='Directory containing preprocessed data')
    parser.add_argument('--train_dir', type=str, default=None, help='Directory containing training data (for split-based workflow)')
    parser.add_argument('--val_dir', type=str, default=None, help='Directory containing validation data (for split-based workflow)')
    parser.add_argument('--test_dir', type=str, default=None, help='Directory containing test data (for split-based workflow)')
    parser.add_argument('--output_dir', type=str, default='logs/training', help='Directory to save model and results')
    parser.add_argument('--use_split_dirs', type=bool, default=False, help='Whether to use split-based workflow with separate directories')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat'], help='Type of GNN model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_split_dirs:
        if not all([args.train_dir, args.val_dir, args.test_dir]):
            parser.error("When using split-based workflow, train_dir, val_dir, and test_dir must be provided")
    else:
        if not args.preprocessed_dir:
            parser.error("When using standard workflow, preprocessed_dir must be provided")
    
    results = train_model(
        preprocessed_dir=args.preprocessed_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        use_split_dirs=args.use_split_dirs,
    )
