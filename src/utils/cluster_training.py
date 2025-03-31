import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.gnn import GNNRegressor
from src.utils.visualization import VisualizationManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ClusterBasedTrainer:
    """
    Trainer for cluster-based GNN models.
    
    Trains separate GNN models for each cluster of data.
    """
    
    def __init__(
        self,
        n_clusters: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = 'gcn',
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        resume_checkpoint: Optional[str] = None
    ):
        """
        Initialize the cluster-based trainer.
        
        Args:
            n_clusters (int): Number of clusters
            hidden_dim (int): Hidden dimension size
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate
            model_type (str): Type of GNN ('gcn' or 'gat')
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for optimizer
            batch_size (int): Batch size
            epochs (int): Number of epochs
            early_stopping_patience (int): Patience for early stopping
            device (torch.device, optional): Device to use
            output_dir (str, optional): Directory to save outputs
            checkpoint_dir (str, optional): Directory to save checkpoints
            checkpoint_interval (int): Interval for saving checkpoints
            resume_checkpoint (str, optional): Path to checkpoint to resume from
        """
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.resume_checkpoint = resume_checkpoint
        
        # Initialize models, optimizers, and other attributes
        self.models = {}
        self.optimizers = {}
        self.best_val_losses = {}
        self.train_losses = {i: [] for i in range(n_clusters)}
        self.val_losses = {i: [] for i in range(n_clusters)}
        self.best_epochs = {}
        
        # Initialize visualization manager
        if output_dir is not None:
            self.viz_manager = VisualizationManager(output_dir, "cluster_training")
        else:
            self.viz_manager = None
    
    def _create_model(self, input_dim: int, cluster_id: int) -> GNNRegressor:
        """
        Create a GNN model for a specific cluster.
        
        Args:
            input_dim (int): Input dimension
            cluster_id (int): Cluster ID
            
        Returns:
            GNNRegressor: GNN model
        """
        model = GNNRegressor(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type=self.model_type
        ).to(self.device)
        
        return model
    
    def _create_optimizer(self, model: GNNRegressor) -> optim.Optimizer:
        """
        Create an optimizer for a model.
        
        Args:
            model (GNNRegressor): GNN model
            
        Returns:
            optim.Optimizer: Optimizer
        """
        return optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _train_epoch(
        self, 
        model: GNNRegressor, 
        optimizer: optim.Optimizer, 
        train_loader: DataLoader
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            model (GNNRegressor): GNN model
            optimizer (optim.Optimizer): Optimizer
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Training loss
        """
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = nn.MSELoss()(out, data.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
        
        return total_loss / len(train_loader.dataset)
    
    def _validate(self, model: GNNRegressor, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            model (GNNRegressor): GNN model
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Validation loss
        """
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = nn.MSELoss()(out, data.y)
                total_loss += loss.item() * data.num_graphs
        
        return total_loss / len(val_loader.dataset)
    
    def _test(self, model: GNNRegressor, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.
        
        Args:
            model (GNNRegressor): GNN model
            test_loader (DataLoader): Test data loader
            
        Returns:
            Dict[str, float]: Test metrics
        """
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                predictions.append(out.cpu().numpy())
                targets.append(data.y.cpu().numpy())
        
        # Concatenate predictions and targets
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
    
    def train_cluster_models(
        self,
        train_loaders: Dict[int, DataLoader],
        val_loaders: Dict[int, DataLoader],
        test_loaders: Dict[int, DataLoader]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Train models for each cluster.
        
        Args:
            train_loaders (Dict[int, DataLoader]): Training data loaders for each cluster
            val_loaders (Dict[int, DataLoader]): Validation data loaders for each cluster
            test_loaders (Dict[int, DataLoader]): Test data loaders for each cluster
            
        Returns:
            Dict[int, Dict[str, Any]]: Test results for each cluster
        """
        logger.info(f"Training models for {len(train_loaders)} clusters")
        
        # Check if we need to resume from checkpoint
        if self.resume_checkpoint is not None:
            self._resume_from_checkpoint()
        
        # Train models for each cluster
        for cluster_id, train_loader in train_loaders.items():
            logger.info(f"Training model for cluster {cluster_id}")
            
            # Get data loaders
            val_loader = val_loaders[cluster_id]
            test_loader = test_loaders[cluster_id]
            
            # Create model and optimizer if not already created
            if cluster_id not in self.models:
                # Get input dimension from first batch
                for data in train_loader:
                    input_dim = data.x.size(1)
                    break
                
                self.models[cluster_id] = self._create_model(input_dim, cluster_id)
                self.optimizers[cluster_id] = self._create_optimizer(self.models[cluster_id])
                self.best_val_losses[cluster_id] = float('inf')
                self.best_epochs[cluster_id] = -1
            
            # Train the model
            start_time = time.time()
            
            for epoch in range(self.epochs):
                # Train for one epoch
                train_loss = self._train_epoch(
                    self.models[cluster_id],
                    self.optimizers[cluster_id],
                    train_loader
                )
                
                # Validate
                val_loss = self._validate(self.models[cluster_id], val_loader)
                
                # Save losses
                self.train_losses[cluster_id].append(train_loss)
                self.val_losses[cluster_id].append(val_loss)
                
                # Check for best model
                if val_loss < self.best_val_losses[cluster_id]:
                    self.best_val_losses[cluster_id] = val_loss
                    self.best_epochs[cluster_id] = epoch
                    
                    # Save best model
                    if self.checkpoint_dir is not None:
                        self.models[cluster_id].save_checkpoint(
                            self.checkpoint_dir,
                            epoch,
                            self.optimizers[cluster_id],
                            train_loss,
                            val_loss,
                            is_best=True,
                            filename=f"best_model_cluster_{cluster_id}.pt"
                        )
                
                # Save checkpoint at regular intervals
                if self.checkpoint_dir is not None and (epoch + 1) % self.checkpoint_interval == 0:
                    self.models[cluster_id].save_checkpoint(
                        self.checkpoint_dir,
                        epoch,
                        self.optimizers[cluster_id],
                        train_loss,
                        val_loss,
                        filename=f"checkpoint_epoch_{epoch}_cluster_{cluster_id}.pt"
                    )
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Cluster {cluster_id}, Epoch {epoch+1}/{self.epochs}, "
                                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if epoch - self.best_epochs[cluster_id] >= self.early_stopping_patience:
                    logger.info(f"Early stopping for cluster {cluster_id} at epoch {epoch+1}")
                    break
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Load best model for testing
            if self.checkpoint_dir is not None:
                best_model_path = os.path.join(self.checkpoint_dir, f"best_model_cluster_{cluster_id}.pt")
                if os.path.exists(best_model_path):
                    self.models[cluster_id], _, _ = GNNRegressor.load_checkpoint(best_model_path, self.device)
            
            # Test the model
            test_results = self._test(self.models[cluster_id], test_loader)
            test_results['training_time'] = training_time
            test_results['best_epoch'] = self.best_epochs[cluster_id]
            test_results['best_val_loss'] = self.best_val_losses[cluster_id]
            
            # Save training curves
            if self.viz_manager is not None:
                self.viz_manager.plot_training_curves(
                    self.train_losses[cluster_id],
                    self.val_losses[cluster_id],
                    self.best_epochs[cluster_id],
                    title=f"Training and Validation Loss - Cluster {cluster_id}"
                )
                
                self.viz_manager.plot_predictions_vs_targets(
                    test_results['predictions'],
                    test_results['targets'],
                    title=f"Predictions vs Targets - Cluster {cluster_id}"
                )
                
                self.viz_manager.plot_error_distribution(
                    test_results['predictions'],
                    test_results['targets'],
                    title=f"Prediction Error Distribution - Cluster {cluster_id}"
                )
            
            # Save test results
            if self.output_dir is not None:
                results_path = os.path.join(self.output_dir, f"test_results_cluster_{cluster_id}.pkl")
                with open(results_path, 'wb') as f:
                    pickle.dump(test_results, f)
            
            logger.info(f"Cluster {cluster_id} - Test RMSE: {test_results['rmse']:.4f}, "
                        f"MAE: {test_results['mae']:.4f}, R²: {test_results['r2']:.4f}")
        
        # Combine results from all clusters
        combined_results = self._combine_cluster_results(test_loaders)
        
        return combined_results
    
    def _combine_cluster_results(self, test_loaders: Dict[int, DataLoader]) -> Dict[int, Dict[str, Any]]:
        """
        Combine results from all clusters.
        
        Args:
            test_loaders (Dict[int, DataLoader]): Test data loaders for each cluster
            
        Returns:
            Dict[int, Dict[str, Any]]: Combined test results
        """
        # Get all predictions and targets
        all_predictions = []
        all_targets = []
        
        for cluster_id, model in self.models.items():
            test_loader = test_loaders[cluster_id]
            
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    all_predictions.append(out.cpu().numpy())
                    all_targets.append(data.y.cpu().numpy())
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        combined_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        # Save combined results
        if self.output_dir is not None:
            results_path = os.path.join(self.output_dir, "combined_test_results.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(combined_results, f)
            
            # Save combined visualizations
            if self.viz_manager is not None:
                self.viz_manager.plot_predictions_vs_targets(
                    all_predictions,
                    all_targets,
                    title="Combined Predictions vs Targets"
                )
                
                self.viz_manager.plot_error_distribution(
                    all_predictions,
                    all_targets,
                    title="Combined Prediction Error Distribution"
                )
        
        logger.info(f"Combined - Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return combined_results
    
    def _resume_from_checkpoint(self):
        """
        Resume training from checkpoint.
        """
        logger.info(f"Resuming from checkpoint: {self.resume_checkpoint}")
        
        # Check if checkpoint is a directory or a file
        if os.path.isdir(self.resume_checkpoint):
            # Find latest checkpoint for each cluster
            for cluster_id in range(self.n_clusters):
                latest_checkpoint = GNNRegressor.find_latest_checkpoint(
                    os.path.join(self.resume_checkpoint, f"cluster_{cluster_id}")
                )
                
                if latest_checkpoint is not None:
                    model, optimizer_state, epoch = GNNRegressor.load_checkpoint(latest_checkpoint, self.device)
                    
                    self.models[cluster_id] = model
                    self.optimizers[cluster_id] = self._create_optimizer(model)
                    self.optimizers[cluster_id].load_state_dict(optimizer_state)
                    
                    logger.info(f"Resumed cluster {cluster_id} from epoch {epoch}")
        else:
            # Load specific checkpoint
            model, optimizer_state, epoch = GNNRegressor.load_checkpoint(self.resume_checkpoint, self.device)
            
            # Extract cluster ID from filename
            filename = os.path.basename(self.resume_checkpoint)
            if "cluster_" in filename:
                cluster_id = int(filename.split("cluster_")[1].split(".")[0])
                
                self.models[cluster_id] = model
                self.optimizers[cluster_id] = self._create_optimizer(model)
                self.optimizers[cluster_id].load_state_dict(optimizer_state)
                
                logger.info(f"Resumed cluster {cluster_id} from epoch {epoch}")
    
    def save_models(self, output_dir: Optional[str] = None):
        """
        Save all models.
        
        Args:
            output_dir (str, optional): Directory to save models
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        if output_dir is None:
            logger.warning("No output directory specified, models not saved")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for cluster_id, model in self.models.items():
            model_path = os.path.join(output_dir, f"model_cluster_{cluster_id}.pt")
            torch.save(model.state_dict(), model_path)
        
        logger.info(f"Models saved to {output_dir}")
    
    def load_models(self, input_dir: str):
        """
        Load all models.
        
        Args:
            input_dir (str): Directory with saved models
        """
        for cluster_id in range(self.n_clusters):
            model_path = os.path.join(input_dir, f"model_cluster_{cluster_id}.pt")
            
            if os.path.exists(model_path):
                # Create model
                input_dim = 1  # Default, will be overwritten when loading state dict
                model = self._create_model(input_dim, cluster_id)
                
                # Load state dict
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                
                self.models[cluster_id] = model
        
        logger.info(f"Models loaded from {input_dir}")
