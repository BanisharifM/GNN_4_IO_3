import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data import IOCounterGraph
from src.models.gnn import GNNRegressor
from src.utils.shap_utils import GNNExplainer

def test_shap_analysis():
    """
    Test the SHAP analysis module by:
    1. Loading a trained GNN model
    2. Creating a test dataset
    3. Computing SHAP values for node features
    4. Identifying I/O bottlenecks
    5. Visualizing the results
    """
    print("Testing SHAP analysis module...")
    
    # Set paths
    data_dir = os.path.join(project_root, "data")
    data_file = os.path.join(data_dir, "sample_train_100.csv")
    mi_file = os.path.join(data_dir, "mutual_information2.csv")
    model_path = os.path.join(project_root, "logs", "training", "gnn_model.pt")
    output_dir = os.path.join(project_root, "logs", "shap")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists, if not, train a simple model for testing
    if not os.path.exists(model_path):
        print("Trained model not found. Training a simple model for testing...")
        from test_gnn_training import test_gnn_model_training
        model, dataset = test_gnn_model_training()
    else:
        print("Loading trained model...")
        # Load model and dataset
        model = load_model(model_path)
        dataset = create_dataset(data_file, mi_file)
    
    # Get counter names
    graph_constructor = IOCounterGraph()
    mi_df = graph_constructor.load_mutual_information(mi_file)
    graph_constructor.construct_graph(mi_df)
    counter_names = graph_constructor.counter_names
    
    # Initialize explainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explainer = GNNExplainer(model, device)
    
    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values, _ = explainer.compute_node_shap_values(
        data_list=dataset,
        counter_names=counter_names,
        background_samples=10,  # Using a small number for testing
        test_samples=5,         # Using a small number for testing
        save_dir=output_dir
    )
    
    # Identify bottlenecks
    print("Identifying I/O bottlenecks...")
    bottlenecks_df = explainer.identify_bottlenecks(
        shap_values=shap_values,
        counter_names=counter_names,
        top_n=10,
        save_dir=output_dir
    )
    
    # Print top bottlenecks
    print("\nTop 10 I/O Bottlenecks:")
    for i, (counter, importance) in enumerate(zip(bottlenecks_df['counter'][:10], bottlenecks_df['importance'][:10])):
        print(f"{i+1}. {counter}: {importance:.4f}")
    
    print(f"\nSHAP analysis test completed. Results saved to {output_dir}")
    
    return bottlenecks_df

def load_model(model_path):
    """
    Load a trained GNN model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        GNNRegressor: Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNRegressor(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        model_type='gcn'
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def create_dataset(data_file, mi_file, mi_threshold=0.3259):
    """
    Create a dataset for SHAP analysis.
    
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
    test_shap_analysis()
