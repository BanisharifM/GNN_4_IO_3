#!/usr/bin/env python
# Step 3: Perform SHAP analysis for bottleneck identification

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pickle
import pandas as pd
import shap
from torch_geometric.data import Data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from src.models.gnn import GNNRegressor
from src.utils.shap_utils import GNNExplainer

def analyze_bottlenecks(
    preprocessed_dir,
    model_dir,
    output_dir,
    background_samples=50,
    test_samples=20,
    top_n=10,
    device=None
):
    """
    Perform SHAP analysis to identify I/O bottlenecks.
    
    Args:
        preprocessed_dir (str): Directory with preprocessed data
        model_dir (str): Directory with trained model
        output_dir (str): Directory to save SHAP analysis results
        background_samples (int): Number of background samples for SHAP
        test_samples (int): Number of test samples to explain
        top_n (int): Number of top bottlenecks to identify
        device (str): Device to use ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Load preprocessed data
    logger.info(f"Loading preprocessed data from {preprocessed_dir}")
    
    edge_index = torch.load(os.path.join(preprocessed_dir, 'edge_index.pt'))
    edge_attr = torch.load(os.path.join(preprocessed_dir, 'edge_attr.pt'))
    node_features = torch.load(os.path.join(preprocessed_dir, 'node_features.pt'))
    targets = torch.load(os.path.join(preprocessed_dir, 'targets.pt'))
    
    # Load counter names
    with open(os.path.join(preprocessed_dir, 'counter_names.pkl'), 'rb') as f:
        counter_names = pickle.load(f)
    
    # Load test indices
    test_indices = np.load(os.path.join(preprocessed_dir, 'test_indices.npy'))
    
    # Load model hyperparameters
    with open(os.path.join(model_dir, 'training_results.pkl'), 'rb') as f:
        training_results = pickle.load(f)
    
    hyperparams = training_results['hyperparameters']
    
    # Create model
    logger.info(f"Creating model with hyperparameters: {hyperparams}")
    
    model = GNNRegressor(
        input_dim=1,
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        model_type=hyperparams['model_type']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device))
    model.eval()
    
    # Create dataset for SHAP analysis
    logger.info("Creating dataset for SHAP analysis")
    
    # Use test set for SHAP analysis
    test_node_features = node_features[test_indices]
    test_targets = targets[test_indices]
    
    # Create data objects
    data_list = []
    for i in range(len(test_node_features)):
        data = Data(
            x=test_node_features[i].view(-1, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=test_targets[i].view(-1)
        )
        data_list.append(data)
    
    # Initialize explainer
    logger.info("Initializing GNN explainer")
    explainer = GNNExplainer(model, device)
    
    # Compute SHAP values
    logger.info(f"Computing SHAP values with {background_samples} background samples and {test_samples} test samples")
    shap_values, _ = explainer.compute_node_shap_values(
        data_list=data_list,
        counter_names=counter_names,
        background_samples=background_samples,
        test_samples=test_samples,
        save_dir=output_dir
    )
    
    # Identify bottlenecks
    logger.info(f"Identifying top {top_n} I/O bottlenecks")
    bottlenecks_df = explainer.identify_bottlenecks(
        shap_values=shap_values,
        counter_names=counter_names,
        top_n=top_n,
        save_dir=output_dir
    )
    
    # Print top bottlenecks
    logger.info("\nTop I/O Bottlenecks:")
    for i, (counter, importance) in enumerate(zip(bottlenecks_df['counter'][:top_n], bottlenecks_df['importance'][:top_n])):
        logger.info(f"{i+1}. {counter}: {importance:.4f}")
    
    # Save bottlenecks to CSV
    bottlenecks_df.to_csv(os.path.join(output_dir, 'bottlenecks.csv'), index=False)
    
    # Create additional visualizations
    
    # Bar plot of top bottlenecks
    plt.figure(figsize=(12, 8))
    plt.barh(bottlenecks_df['counter'][:top_n][::-1], bottlenecks_df['importance'][:top_n][::-1])
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {top_n} I/O Bottlenecks')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_bottlenecks.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pie chart of top bottlenecks
    plt.figure(figsize=(10, 10))
    plt.pie(
        bottlenecks_df['importance'][:top_n],
        labels=bottlenecks_df['counter'][:top_n],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.axis('equal')
    plt.title(f'Relative Importance of Top {top_n} I/O Bottlenecks')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bottlenecks_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP analysis completed. Results saved to {output_dir}")
    
    return bottlenecks_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform SHAP analysis for bottleneck identification")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save SHAP analysis results")
    parser.add_argument("--background_samples", type=int, default=50, help="Number of background samples for SHAP")
    parser.add_argument("--test_samples", type=int, default=20, help="Number of test samples to explain")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top bottlenecks to identify")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    bottlenecks_df = analyze_bottlenecks(
        args.preprocessed_dir,
        args.model_dir,
        args.output_dir,
        args.background_samples,
        args.test_samples,
        args.top_n,
        args.device
    )
    
    print("\nTop I/O Bottlenecks:")
    for i, (counter, importance) in enumerate(zip(bottlenecks_df['counter'][:args.top_n], bottlenecks_df['importance'][:args.top_n])):
        print(f"{i+1}. {counter}: {importance:.4f}")
