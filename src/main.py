import os
import hydra
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from data import IOCounterGraph, IOGraphDataset
from models.gnn import GNNRegressor
from utils.shap_utils import GNNExplainer, analyze_bottlenecks
from utils.data_utils import get_file_path

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run the GNN-based I/O bottleneck analysis.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    logger.info(f"Running with configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set paths
    data_path = cfg.data.path if hasattr(cfg.data, 'path') else os.path.join(os.getcwd(), '..', 'data')
    data_file = os.path.join(data_path, cfg.data.files.sample_100)
    mi_file = os.path.join(data_path, cfg.data.files.mutual_information)
    
    # Create directories
    os.makedirs(cfg.model.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.logs.training_dir, exist_ok=True)
    os.makedirs(cfg.logs.shap_dir, exist_ok=True)
    
    # Load dataset
    logger.info("Creating dataset...")
    dataset = create_dataset(data_file, mi_file, cfg.graph.mi_threshold)
    
    # Split dataset
    train_size = int(cfg.data.train_split * len(dataset))
    val_size = int(cfg.data.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size)
    
    # Create model
    logger.info("Creating GNN model...")
    model = GNNRegressor(
        input_dim=1,  # Single feature per node
        hidden_dim=cfg.gnn.hidden_dim,
        num_layers=cfg.gnn.num_layers,
        dropout=cfg.hyperparameters.dropout,
        model_type=cfg.gnn.model_type
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.learning_rate)
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(cfg.hyperparameters.epochs):
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
        
        logger.info(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.model.checkpoint_dir, 'best_model.pt'))
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(cfg.model.checkpoint_dir, 'best_model.pt')))
    
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
    plt.plot(range(1, cfg.hyperparameters.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, cfg.hyperparameters.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.logs.training_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    plt.savefig(os.path.join(cfg.logs.training_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Get counter names
    graph_constructor = IOCounterGraph()
    mi_df = graph_constructor.load_mutual_information(mi_file)
    graph_constructor.construct_graph(mi_df)
    counter_names = graph_constructor.counter_names
    
    # SHAP analysis
    logger.info("Performing SHAP analysis for bottleneck identification...")
    bottlenecks_df = analyze_bottlenecks(
        model=model,
        dataset=dataset,
        counter_names=counter_names,
        cfg=cfg
    )
    
    # Print top bottlenecks
    logger.info("\nTop I/O Bottlenecks:")
    for i, (counter, importance) in enumerate(zip(bottlenecks_df['counter'][:10], bottlenecks_df['importance'][:10])):
        logger.info(f"{i+1}. {counter}: {importance:.4f}")
    
    logger.info(f"Analysis completed. Results saved to {cfg.logs.training_dir} and {cfg.logs.shap_dir}")

def create_dataset(data_file, mi_file, mi_threshold=0.3259):
    """
    Create a dataset for GNN training and SHAP analysis.
    
    Args:
        data_file (str): Path to the data CSV file
        mi_file (str): Path to the mutual information CSV file
        mi_threshold (float): Threshold for mutual information to create an edge
        
    Returns:
        list: List of PyTorch Geometric Data objects
    """
    # Load data
    data_df = pd.read_csv(data_file)
    
    # Initialize graph constructor
    graph_constructor = IOCounterGraph(mi_threshold=mi_threshold)
    
    # Load mutual information and construct graph structure
    mi_df = graph_constructor.load_mutual_information(mi_file)
    edge_index, edge_attr = graph_constructor.construct_graph(mi_df)
    
    # Extract node features and targets
    node_features = graph_constructor.get_node_features(data_df)
    targets = torch.tensor(data_df['tag'].values, dtype=torch.float)
    
    # Create data list
    data_list = []
    
    for i in range(len(data_df)):
        # Create PyTorch Geometric Data object
        data = torch.utils.data.dataset.Dataset.__new__(torch.utils.data.dataset.Dataset)
        data.x = node_features[i].view(-1, 1)  # [num_nodes, 1]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.y = targets[i].view(-1)  # [1]
        data.num_nodes = len(graph_constructor.counter_names)
        
        data_list.append(data)
    
    return data_list

if __name__ == "__main__":
    main()
