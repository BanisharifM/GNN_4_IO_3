#!/usr/bin/env python
# Step 4: Hyperparameter optimization with Ray Tune - Split Workflow Version

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
from ray.tune.logger import TBXLoggerCallback
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import sklearn.metrics as metrics

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
    train_dir=None,
    val_dir=None,
    test_dir=None,
    checkpoint_dir=None,
    num_epochs=100,
    early_stopping_patience=10
):
    """
    Training function for Ray Tune with split-based workflow.
    
    Args:
        config (dict): Hyperparameters from Ray Tune
        train_dir (str): Directory with training data
        val_dir (str): Directory with validation data
        test_dir (str): Directory with test data
        checkpoint_dir (str): Directory for checkpoints
        num_epochs (int): Maximum number of epochs
        early_stopping_patience (int): Patience for early stopping
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load edge structure (should be the same for all splits, so we use train_dir)
    edge_index = torch.load(os.path.join(train_dir, 'edge_index.pt'))
    edge_attr = torch.load(os.path.join(train_dir, 'edge_attr.pt'))
    
    # Load train data
    train_node_features = torch.load(os.path.join(train_dir, 'node_features.pt'))
    train_targets = torch.load(os.path.join(train_dir, 'targets.pt'))
    
    # Load val data
    val_node_features = torch.load(os.path.join(val_dir, 'node_features.pt'))
    val_targets = torch.load(os.path.join(val_dir, 'targets.pt'))
    
    # Create datasets
    train_dataset = create_dataset(train_node_features, train_targets, edge_index, edge_attr)
    val_dataset = create_dataset(val_node_features, val_targets, edge_index, edge_attr)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']))
    
    # Create model
    model = GNNRegressor(
        input_dim=1,
        hidden_dim=int(config['hidden_dim']),
        output_dim=1,
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
            loss = criterion(out.view(-1), batch.y.view(-1))
 
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch.num_graphs
        
        train_loss = epoch_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out.view(-1), batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs
                
                # Store predictions and targets for metrics
                val_preds.extend(out.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate additional metrics
        val_rmse = np.sqrt(metrics.mean_squared_error(val_targets, val_preds))
        val_mae = metrics.mean_absolute_error(val_targets, val_preds)
        val_r2 = metrics.r2_score(val_targets, val_preds)
        
        # Report metrics to Ray Tune
        tune.report({
            "loss": val_loss,
            "train_loss": train_loss,
            "rmse": val_rmse,
            "mae": val_mae,
            "r2": val_r2,
            "epoch": epoch
        })
  
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
    train_dir,
    val_dir,
    test_dir,
    output_dir,
    num_samples=10,
    max_epochs=100,
    early_stopping_patience=10,
    cpus_per_trial=1,
    gpus_per_trial=0.5,
    num_workers=2
):
    """
    Optimize hyperparameters using Ray Tune with split-based workflow.
    
    Args:
        train_dir (str): Directory with training data
        val_dir (str): Directory with validation data
        test_dir (str): Directory with test data
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
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        num_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience
    )
    
    # Set up TensorBoard callback
    tb_callback = TBXLoggerCallback() 
    
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
        storage_path=os.path.abspath(output_dir),
        name="gnn_hyperopt",
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-loss",
        progress_reporter=tune.CLIReporter(
            metric_columns=["loss", "train_loss", "rmse", "mae", "r2", "epoch", "training_iteration"]
        ),
        callbacks=[tb_callback]
    )
    
    optimization_time = time.time() - start_time
    
    # Get best configuration
    best_config = analysis.get_best_config(metric="loss", mode="min")
    best_trial = analysis.get_best_trial(metric="loss", mode="min")
    best_loss = best_trial.last_result["loss"]
    best_rmse = best_trial.last_result.get("rmse", "N/A")
    best_mae = best_trial.last_result.get("mae", "N/A")
    best_r2 = best_trial.last_result.get("r2", "N/A")
    
    logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Best validation RMSE: {best_rmse}")
    logger.info(f"Best validation MAE: {best_mae}")
    logger.info(f"Best validation R²: {best_r2}")
    logger.info(f"Best hyperparameters: {best_config}")
    
    # Save results
    results = {
        "best_config": best_config,
        "best_metrics": {
            "loss": best_loss,
            "rmse": best_rmse,
            "mae": best_mae,
            "r2": best_r2
        },
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
        f.write(f"python -W ignore {os.path.join(project_root, 'scripts', '02_train_model_single_checkpoint.py')} \\\n")
        f.write(f"  --train_dir {train_dir} \\\n")
        f.write(f"  --val_dir {val_dir} \\\n")
        f.write(f"  --test_dir {test_dir} \\\n")
        f.write(f"  --output_dir {os.path.join(output_dir, 'best_model')} \\\n")
        f.write(f"  --hidden_dim {int(best_config['hidden_dim'])} \\\n")
        f.write(f"  --num_layers {int(best_config['num_layers'])} \\\n")
        f.write(f"  --model_type {best_config['model_type']} \\\n")
        f.write(f"  --learning_rate {best_config['learning_rate']} \\\n")
        f.write(f"  --batch_size {int(best_config['batch_size'])} \\\n")
        f.write(f"  --dropout {best_config['dropout']} \\\n")
        f.write(f"  --epochs 100 \\\n")
        f.write(f"  --early_stopping_patience 10 \\\n")
        f.write(f"  --use_split_dirs True \\\n")
        f.write(f"  --resume_training True\n")
    
    # Make script executable
    os.chmod(best_params_script, 0o755)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Script to train with best hyperparameters: {best_params_script}")
    logger.info(f"TensorBoard logs available at: {os.path.join(output_dir, 'tensorboard')}")
    logger.info(f"View TensorBoard with: tensorboard --logdir={os.path.join(output_dir, 'tensorboard')}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Ray Tune - Split Workflow")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training data")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory with validation data")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save optimization results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of hyperparameter configurations to try")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs per trial")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="CPUs per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=0.5, help="GPUs per trial")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    results = optimize_hyperparameters(
        os.path.abspath(args.train_dir),
        os.path.abspath(args.val_dir),
        os.path.abspath(args.test_dir),
        os.path.abspath(args.output_dir),
        args.num_samples,
        args.max_epochs,
        args.early_stopping_patience,
        args.cpus_per_trial,
        args.gpus_per_trial,
        args.num_workers
    )
    
    print("\nHyperparameter Optimization Results:")
    print(f"Best validation loss: {results['best_metrics']['loss']:.4f}")
    print(f"Best validation RMSE: {results['best_metrics']['rmse']}")
    print(f"Best validation MAE: {results['best_metrics']['mae']}")
    print(f"Best validation R²: {results['best_metrics']['r2']}")
    print("Best hyperparameters:")
    for param, value in results['best_config'].items():
        print(f"  {param}: {value}")
