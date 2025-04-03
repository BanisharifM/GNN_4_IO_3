#!/usr/bin/env python
# Step 2: Train GNN model with resume functionality and single checkpoint

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pickle
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
    epochs=15,
    dropout=0.1,
    early_stopping_patience=10,
    device=None,
    use_split_dirs=False,
    resume_training=True
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
        
        # Create datasets
        logger.info("Creating datasets from split directories...")
        train_dataset = create_dataset(train_node_features, train_targets, edge_index, edge_attr)
        val_dataset = create_dataset(val_node_features, val_targets, edge_index, edge_attr)
        test_dataset = create_dataset(test_node_features, test_targets, edge_index, edge_attr)
        
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
        
        train_dataset = create_dataset(node_features[train_indices], targets[train_indices], edge_index, edge_attr)
        val_dataset = create_dataset(node_features[val_indices], targets[val_indices], edge_index, edge_attr)
        test_dataset = create_dataset(node_features[test_indices], targets[test_indices], edge_index, edge_attr)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    
    # Create model
    logger.info(f"Creating {model_type.upper()} model with {num_layers} layers and {hidden_dim} hidden dimensions")
    
    model = GNNRegressor(
        input_dim=1,  # Single feature per node
        hidden_dim=hidden_dim,
        output_dim=1,  # Single target value
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type  # This matches the parameter name in GNNRegressor
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
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
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
                'input_dim': 1,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'model_type': model_type
            }
        }
        
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model (separate file)
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} for evaluation")
    
    # Test evaluation
    logger.info("Evaluating model on test set...")
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            test_loss += loss.item() * batch.num_graphs
            
            # Store predictions and targets
            predictions.extend(out.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
    logger.info(f"Test RMSE: {rmse:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch + 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_rmse': rmse,
        'training_time': training_time,
        'hyperparameters': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'model_type': model_type,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout': dropout
        }
    }
    
    # Save results as pickle
    with open(os.path.join(output_dir, 'training_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Training results saved to {output_dir}")
    
    return results

def create_dataset(node_features, targets, edge_index, edge_attr):
    """
    Create a PyTorch Geometric dataset from preprocessed data.
    
    Args:
        node_features (torch.Tensor): Node features [num_samples, num_nodes]
        targets (torch.Tensor): Target values [num_samples]
        edge_index (torch.Tensor): Edge indices [2, num_edges]
        edge_attr (torch.Tensor): Edge attributes [num_edges, edge_dim]
        
    Returns:
        list: List of PyTorch Geometric Data objects
    """
    data_list = []
    
    for i in range(len(node_features)):
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features[i].view(-1, 1),  # [num_nodes, 1]
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=targets[i].view(-1)  # [1]
        )
        
        data_list.append(data)
    
    return data_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN model")
    
    # Data parameters - support both workflows
    parser.add_argument("--preprocessed_dir", type=str, default=None, help="Directory with preprocessed data (for standard workflow)")
    parser.add_argument("--train_dir", type=str, default=None, help="Directory with training data (for split-based workflow)")
    parser.add_argument("--val_dir", type=str, default=None, help="Directory with validation data (for split-based workflow)")
    parser.add_argument("--test_dir", type=str, default=None, help="Directory with test data (for split-based workflow)")
    parser.add_argument("--use_split_dirs", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to use split-based workflow")
    parser.add_argument("--output_dir", type=str, default='logs/training', help="Directory to save model and results")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--model_type", type=str, default='gcn', choices=['gcn', 'gat'], help="Type of GNN")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--resume_training", type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="Whether to resume training from latest checkpoint if available")
    
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
        resume_training=args.resume_training
    )
    
    print("\nTraining Results:")
    print(f"Best Epoch: {results['best_epoch']+1}")
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")

