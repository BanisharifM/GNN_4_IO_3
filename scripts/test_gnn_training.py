import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data import IOCounterGraph, IOGraphDataset
from src.models.gnn import GNNModel, GNNRegressor

def test_gnn_model_training():
    """
    Test the GNN model training by:
    1. Loading the dataset
    2. Creating a GNN model
    3. Training the model for a few epochs
    4. Evaluating the model performance
    """
    print("Testing GNN model training...")
    
    # Set paths
    data_dir = os.path.join(project_root, "data")
    data_file = os.path.join(data_dir, "sample_train_100.csv")
    mi_file = os.path.join(data_dir, "mutual_information2.csv")
    output_dir = os.path.join(project_root, "logs", "training")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set parameters
    mi_threshold = 0.3259
    hidden_dim = 64
    num_layers = 2
    learning_rate = 0.001
    batch_size = 16
    num_epochs = 10
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(data_file, mi_file, mi_threshold)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    print("Creating GNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNRegressor(
        input_dim=1,  # Single feature per node
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        model_type='gcn'
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Test evaluation
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
    print(f"Test Loss: {test_loss:.4f}")
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
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
    
    print(f"GNN model training test completed. Results saved to {output_dir}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'gnn_model.pt'))
    print(f"Model saved to {os.path.join(output_dir, 'gnn_model.pt')}")
    
    return model, dataset

def create_dataset(data_file, mi_file, mi_threshold):
    """
    Create a PyTorch Geometric dataset from the data files.
    
    Args:
        data_file (str): Path to the data CSV file
        mi_file (str): Path to the mutual information CSV file
        mi_threshold (float): Threshold for mutual information to create an edge
        
    Returns:
        IOGraphDataset: PyTorch Geometric dataset
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
    test_gnn_model_training()
