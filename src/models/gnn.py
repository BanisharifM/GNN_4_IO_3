import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GNNModel(nn.Module):
    """
    Base GNN model with configurable architecture.
    
    Supports GCN and GAT architectures.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int,
        dropout: float = 0.1,
        model_type: str = 'gcn'
    ):
        """
        Initialize the GNN model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension size
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate
            model_type (str): Type of GNN ('gcn' or 'gat')
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type.lower()
        
        # Input layer
        if self.model_type == 'gcn':
            self.conv_layers = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        elif self.model_type == 'gat':
            self.conv_layers = nn.ModuleList([GATConv(input_dim, hidden_dim)])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if self.model_type == 'gcn':
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif self.model_type == 'gat':
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim))
        
        logger.info(f"Initialized {model_type.upper()} model with {num_layers} layers and {hidden_dim} hidden dimensions")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GNN layers.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor, optional): Edge attributes [num_edges, edge_dim]
            batch (torch.Tensor, optional): Batch indices [num_nodes]
            
        Returns:
            torch.Tensor: Graph-level representation [batch_size, hidden_dim]
        """
        # Process through GNN layers
        for i, conv in enumerate(self.conv_layers):
            if self.model_type == 'gcn' and edge_attr is not None:
                # For GCN, use edge weights if available
                x = conv(x, edge_index, edge_weight=edge_attr.view(-1))
            else:
                # For GAT or if no edge attributes
                x = conv(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GNNRegressor(nn.Module):
    """
    Regression model for predicting performance tags.
    
    Wraps the GNN model with regression output.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int,
        output_dim=1,
        dropout: float = 0.1,
        model_type: str = 'gcn'
    ):
        """
        Initialize the GNN regressor.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension size
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate
            model_type (str): Type of GNN ('gcn' or 'gat')
        """
        super(GNNRegressor, self).__init__()
        
        # GNN model for graph-level representation
        self.gnn = GNNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )
        
        # Regression head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized GNN regressor with {model_type.upper()} backbone")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GNN regressor.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor, optional): Edge attributes [num_edges, edge_dim]
            batch (torch.Tensor, optional): Batch indices [num_nodes]
            
        Returns:
            torch.Tensor: Regression output [batch_size, 1]
        """
        # Get graph-level representation from GNN
        x = self.gnn(x, edge_index, edge_attr, batch)
        
        # Regression head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.output_layer(x)

    
    def save_checkpoint(
        self, 
        checkpoint_dir: str, 
        epoch: int, 
        optimizer: torch.optim.Optimizer,
        train_loss: float,
        val_loss: float,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoint
            epoch (int): Current epoch
            optimizer (torch.optim.Optimizer): Optimizer
            train_loss (float): Training loss
            val_loss (float): Validation loss
            is_best (bool): Whether this is the best model so far
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_config': {
                'input_dim': self.gnn.input_dim,
                'hidden_dim': self.gnn.hidden_dim,
                'num_layers': self.gnn.num_layers,
                'dropout': self.gnn.dropout,
                'model_type': self.gnn.model_type
            }
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model, save a copy
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        return checkpoint_path
    
    @classmethod
    def load_checkpoint(
        cls, 
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> Tuple['GNNRegressor', Dict[str, Any], int]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            device (torch.device, optional): Device to load model to
            
        Returns:
            Tuple[GNNRegressor, Dict, int]: Model, optimizer state dict, and epoch
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model configuration
        model_config = checkpoint['model_config']
        
        # Create model
        model = cls(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            model_type=model_config['model_type']
        ).to(device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {checkpoint_path} (epoch {checkpoint['epoch']})")
        
        return model, checkpoint['optimizer_state_dict'], checkpoint['epoch']
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint in the directory.
        
        Args:
            checkpoint_dir (str): Directory with checkpoints
            
        Returns:
            str: Path to latest checkpoint, or None if no checkpoints found
        """
        if not os.path.exists(checkpoint_dir):
            return None
        
        # Find all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        
        if not checkpoint_files:
            return None
        
        # Extract epoch numbers
        epoch_numbers = []
        for filename in checkpoint_files:
            try:
                epoch = int(filename.split("_")[-1].split(".")[0])
                epoch_numbers.append((epoch, filename))
            except ValueError:
                continue
        
        if not epoch_numbers:
            return None
        
        # Find the latest epoch
        latest_epoch, latest_filename = max(epoch_numbers, key=lambda x: x[0])
        latest_checkpoint = os.path.join(checkpoint_dir, latest_filename)
        
        logger.info(f"Found latest checkpoint: {latest_checkpoint} (epoch {latest_epoch})")
        
        return latest_checkpoint
