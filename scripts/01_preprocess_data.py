#!/usr/bin/env python
# Step 1: Preprocess data and construct graph

import os
import argparse
import pandas as pd
import torch
import numpy as np
import logging
from pathlib import Path
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from src.data import IOCounterGraph

def preprocess_data(data_file, mi_file, output_dir, mi_threshold=0.3259, sample_size=None):
    """
    Preprocess data and construct graph structure.
    
    Args:
        data_file (str): Path to the data CSV file
        mi_file (str): Path to the mutual information CSV file
        output_dir (str): Directory to save preprocessed data
        mi_threshold (float): Threshold for mutual information to create an edge
        sample_size (int, optional): Number of samples to use (for testing with smaller dataset)
    """
    logger.info(f"Preprocessing data from {data_file}")
    logger.info(f"Using mutual information from {mi_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data_df = pd.read_csv(data_file)
    
    # Sample data if sample_size is provided
    if sample_size is not None and sample_size < len(data_df):
        logger.info(f"Sampling {sample_size} rows from dataset")
        data_df = data_df.sample(n=sample_size, random_state=42)
    
    # Initialize graph constructor
    logger.info(f"Constructing graph with MI threshold: {mi_threshold}")
    graph_constructor = IOCounterGraph(mi_threshold=mi_threshold)
    
    # Load mutual information and construct graph structure
    mi_df = graph_constructor.load_mutual_information(mi_file)
    edge_index, edge_attr = graph_constructor.construct_graph(mi_df)
    
    # Extract node features
    logger.info("Extracting node features...")
    node_features = graph_constructor.get_node_features(data_df)
    
    # Extract targets
    targets = torch.tensor(data_df['tag'].values, dtype=torch.float)
    
    # Save preprocessed data
    logger.info(f"Saving preprocessed data to {output_dir}")
    
    # Save graph structure
    torch.save(edge_index, os.path.join(output_dir, 'edge_index.pt'))
    torch.save(edge_attr, os.path.join(output_dir, 'edge_attr.pt'))
    
    # Save node features and targets
    torch.save(node_features, os.path.join(output_dir, 'node_features.pt'))
    torch.save(targets, os.path.join(output_dir, 'targets.pt'))
    
    # Save counter names
    with open(os.path.join(output_dir, 'counter_names.pkl'), 'wb') as f:
        pickle.dump(graph_constructor.counter_names, f)
    
    # Create train/val/test splits
    logger.info("Creating train/val/test splits...")
    indices = np.arange(len(data_df))
    np.random.shuffle(indices)
    
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Save splits
    np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(output_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    
    logger.info(f"Preprocessing completed. Data saved to {output_dir}")
    logger.info(f"Dataset size: {len(data_df)} samples")
    logger.info(f"Train/Val/Test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    
    # Return statistics
    stats = {
        'dataset_size': len(data_df),
        'num_nodes': len(graph_constructor.counter_names),
        'num_edges': edge_attr.shape[0],
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and construct graph")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data CSV file")
    parser.add_argument("--mi_file", type=str, required=True, help="Path to the mutual information CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed data")
    parser.add_argument("--mi_threshold", type=float, default=0.3259, help="Threshold for mutual information")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use (for testing)")
    parser.add_argument("--use_advanced_feature_selection", type=bool, default=False)
    parser.add_argument("--use_clustering", type=bool, default=False)

    
    args = parser.parse_args()
    
    stats = preprocess_data(
        args.data_file,
        args.mi_file,
        args.output_dir,
        args.mi_threshold,
        args.sample_size
    )
    
    print("\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
