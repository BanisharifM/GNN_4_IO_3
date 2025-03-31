#!/usr/bin/env python
# Step 4: Hyperparameter optimization with Ray Tune

import os
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
import pickle
import json
import time
from functools import partial
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from src.models.gnn import GNNRegressor

def train_with_params(
    config,
    preprocessed_dir,
    checkpoint_dir=None,
    num_epochs=100,
    early_stopping_patience=10
):
    """
    Training function for Ray Tune.
    
    Args:
        config (dict): Hyperparameters from Ray Tune
        preprocessed_dir (str): Directory with preprocessed data
        checkpoint_dir (str): Directory for checkpoints
        num_epochs (int): Maximum number of epochs
        early_stopping_patience (int): Patience for early stopping
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load preprocessed data
    edge_index = torch.load(os.path.join(preprocessed_dir, 'edge_index.pt'))
    edge_attr = torch.load(os.path.join(preprocessed_dir, 'edge_attr.pt'))
    node_features = torch.load(os.path.join(preprocessed_dir, 'node_features.pt'))
    targets = torch.load(os.path.join(preprocessed_dir, 'targets.pt'))
    
    # Load train/val/test splits
    train_indices = np.load(os.path.join(preprocessed_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(preprocessed_dir, 'val_indices.npy'))
    
    # Create datasets
    train_dataset = create_dataset(node_features[train_indices], targets[train_indices], edge_index, edge_attr)
    val_dataset = create_dataset(node_features[val_indices], targets[val_indices], edge_index, edge_attr)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']))
    
    # Create model
    model = GNNRegressor(
        input_dim=1,
        hidden_dim=int(config['hidden_dim']),
        num_layers=int(config['num_layers']),
        dropout=config['dropout'],
        model_type=config['model_type']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Load checkpoint if available
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
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
        
        # Report metrics to Ray Tune
        tune.report(loss=val_loss, train_loss=train_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), checkpoint)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            break

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

def optimize_hyperparameters(
    preprocessed_dir,
    output_dir,
    num_samples=10,
    max_epochs=100,
    early_stopping_patience=10,
    cpus_per_trial=1,
    gpus_per_trial=0.5,
    num_workers=2
):
    """
    Optimize hyperparameters using Ray Tune.
    
    Args:
        preprocessed_dir (str): Directory with preprocessed data
        output_dir (str): Directory to save optimization results
        num_samples (int): Number of hyperparameter configurations to try
        max_epochs (int): Maximum number of epochs per trial
        early_stopping_patience (int): Patience for early stopping
        cpus_per_trial (int): CPUs per trial
        gpus_per_trial (float): GPUs per trial
        num_workers (int): Number of parallel workers
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting hyperparameter optimization with {num_samples} configurations")
    
    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_dim": tune.choice([32, 64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "dropout": tune.uniform(0.0, 0.5),
        "model_type": tune.choice(["gcn", "gat"])
    }
    
    # Define scheduler
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=10,
        reduction_factor=2
    )
    
    # Define search algorithm
    search_alg = OptunaSearch()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers * cpus_per_trial, num_gpus=num_workers * gpus_per_trial)
    
    # Define training function
    train_fn = partial(
        train_with_params,
        preprocessed_dir=preprocessed_dir,
        num_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience
    )
    
    # Run optimization
    start_time = time.time()
    
    analysis = tune.run(
        train_fn,
        config=search_space,
        metric="loss",
        mode="min",
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        local_dir=output_dir,
        name="gnn_hyperopt",
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-loss",
        progress_reporter=tune.CLIReporter(
            metric_columns=["loss", "train_loss", "training_iteration"]
        )
    )
    
    optimization_time = time.time() - start_time
    
    # Get best configuration
    best_config = analysis.get_best_config(metric="loss", mode="min")
    best_trial = analysis.get_best_trial(metric="loss", mode="min")
    best_loss = best_trial.last_result["loss"]
    
    logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Best hyperparameters: {best_config}")
    
    # Save results
    results = {
        "best_config": best_config,
        "best_loss": best_loss,
        "optimization_time": optimization_time,
        "num_samples": num_samples,
        "max_epochs": max_epochs
    }
    
    # Save as JSON
    with open(os.path.join(output_dir, 'hyperopt_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a script to train with best hyperparameters
    best_params_script = os.path.join(output_dir, 'train_best_model.sh')
    with open(best_params_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"python {os.path.join(project_root, 'scripts', '02_train_model.py')} \\\n")
        f.write(f"  --preprocessed_dir {preprocessed_dir} \\\n")
        f.write(f"  --output_dir {os.path.join(output_dir, 'best_model')} \\\n")
        f.write(f"  --hidden_dim {int(best_config['hidden_dim'])} \\\n")
        f.write(f"  --num_layers {int(best_config['num_layers'])} \\\n")
        f.write(f"  --model_type {best_config['model_type']} \\\n")
        f.write(f"  --learning_rate {best_config['learning_rate']} \\\n")
        f.write(f"  --batch_size {int(best_config['batch_size'])} \\\n")
        f.write(f"  --dropout {best_config['dropout']} \\\n")
        f.write(f"  --epochs 100 \\\n")
        f.write(f"  --early_stopping_patience 10\n")
    
    # Make script executable
    os.chmod(best_params_script, 0o755)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Script to train with best hyperparameters: {best_params_script}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Ray Tune")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save optimization results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of hyperparameter configurations to try")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs per trial")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="CPUs per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=0.5, help="GPUs per trial")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    results = optimize_hyperparameters(
        args.preprocessed_dir,
        args.output_dir,
        args.num_samples,
        args.max_epochs,
        args.early_stopping_patience,
        args.cpus_per_trial,
        args.gpus_per_trial,
        args.num_workers
    )
    
    print("\nHyperparameter Optimization Results:")
    print(f"Best validation loss: {results['best_loss']:.4f}")
    print("Best hyperparameters:")
    for param, value in results['best_config'].items():
        print(f"  {param}: {value}")
