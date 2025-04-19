#!/usr/bin/env python
# Step 1: Preprocess data and construct graph

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import json
from typing import Dict, List, Tuple, Optional

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data import IOCounterGraph, IOGraphDataset, ClusteringModel

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for GNN training')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--mi_file', type=str, required=True, help='Path to mutual information CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed data')
    parser.add_argument('--mi_threshold', type=float, default=0.3259, help='Threshold for mutual information')
    parser.add_argument('--use_advanced_feature_selection', type=lambda x: x.lower() == 'true', default=False, 
                        help='Whether to use advanced feature selection')
    parser.add_argument('--top_features', type=int, default=10, 
                        help='Number of top features to select when using advanced feature selection')
    parser.add_argument('--use_clustering', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use clustering')
    parser.add_argument('--num_clusters', type=int, default=4,
                        help='Number of clusters to use when clustering is enabled')
    parser.add_argument('--split_type', type=str, choices=['train', 'val', 'test', 'all'], default='all',
                        help='Type of split for the input file. Use "all" if the file contains all data that needs to be split.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test data')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def preprocess_data(
    data_file: str,
    mi_file: str,
    output_dir: str,
    mi_threshold: float = 0.3259,
    use_advanced_feature_selection: bool = False,
    top_features: int = 10,
    use_clustering: bool = False,
    num_clusters: int = 4,
    split_type: str = 'all',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict:
    """
    Preprocess data for GNN training.
    
    Args:
        data_file: Path to data CSV file
        mi_file: Path to mutual information CSV file
        output_dir: Directory to save preprocessed data
        mi_threshold: Threshold for mutual information
        use_advanced_feature_selection: Whether to use advanced feature selection
        top_features: Number of top features to select when using advanced feature selection
        use_clustering: Whether to use clustering
        num_clusters: Number of clusters to use when clustering is enabled
        split_type: Type of split for the input file ('train', 'val', 'test', or 'all')
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with preprocessing statistics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing data from {data_file}")
    logger.info(f"Using mutual information from {mi_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data_df = pd.read_csv(data_file)
    
    # Construct graph
    logger.info(f"Constructing graph with MI threshold: {mi_threshold}")
    graph_constructor = IOCounterGraph(mi_threshold=mi_threshold)
    mi_df = graph_constructor.load_mutual_information(mi_file)
    
    # Apply advanced feature selection if requested
    if use_advanced_feature_selection:
        logger.info(f"Using advanced feature selection to select top {top_features} features")
        edge_index, edge_attr = graph_constructor.construct_graph(
            mi_df, 
            use_advanced_feature_selection=True,
            target_col='tag',
            top_n=top_features
        )
        
        # Save selected features
        if graph_constructor.selected_features:
            with open(os.path.join(output_dir, 'selected_features.json'), 'w') as f:
                json.dump(graph_constructor.selected_features, f, indent=2)
            logger.info(f"Saved selected features to {os.path.join(output_dir, 'selected_features.json')}")
    else:
        # Use standard graph construction
        edge_index, edge_attr = graph_constructor.construct_graph(mi_df)
    
    # Export graph structure
    graph_constructor.export_graph_structure(output_dir)
    
    # Save counter mapping
    graph_constructor.save_counter_mapping(output_dir)
    
    # Apply clustering if requested
    clustering_model = None
    if use_clustering:
        logger.info(f"Applying clustering with {num_clusters} clusters")
        clustering_model = ClusteringModel(n_clusters=num_clusters)
        
        # Use selected features from feature selection if available
        initial_features = graph_constructor.selected_features if graph_constructor.selected_features else None
        
        # Fit clustering model
        cluster_labels = clustering_model.fit(data_df, target_col='tag', initial_features=initial_features)
        
        # Add cluster labels to data
        data_df['cluster'] = cluster_labels
        
        # Export clustering information
        clustering_model.export_cluster_info(output_dir, data_df, cluster_labels)
        
        logger.info(f"Clustering completed. {num_clusters} clusters created.")
    
    # Extract node features
    logger.info("Extracting node features...")
    node_features = graph_constructor.get_node_features(data_df)
    
    # Save node features tensor for 02_train_model.py
    torch.save(node_features, os.path.join(output_dir, 'node_features.pt'))
    logger.info(f"Saved node features tensor to {os.path.join(output_dir, 'node_features.pt')}")
    
    # Extract targets
    targets = torch.tensor(data_df['tag'].values, dtype=torch.float)
    
    # Save targets tensor for 02_train_model.py
    torch.save(targets, os.path.join(output_dir, 'targets.pt'))
    logger.info(f"Saved targets tensor to {os.path.join(output_dir, 'targets.pt')}")
    
    # Create data list
    data_list = []
    for i in range(len(data_df)):
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features[i].view(-1, 1),  # [num_nodes, 1]
            edge_index=edge_index,  # Same for all samples
            edge_attr=edge_attr,    # Same for all samples
            y=targets[i].view(-1)   # [1]
        )
        
        # Add cluster information if available
        if use_clustering:
            data.cluster = torch.tensor([data_df.iloc[i]['cluster']], dtype=torch.long)
        
        data_list.append(data)
    
    # Save preprocessed data
    logger.info(f"Saving preprocessed data to {output_dir}")
    
    # Handle different split types
    if split_type == 'all':
        # Split data into train/val/test
        logger.info("Creating train/val/test splits...")
        np.random.seed(random_seed)
        indices = np.random.permutation(len(data_list))
        
        train_size = int(train_ratio * len(data_list))
        val_size = int(val_ratio * len(data_list))
        test_size = len(data_list) - train_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]
        
        # Save splits
        torch.save(train_data, os.path.join(output_dir, 'train_data.pt'))
        torch.save(val_data, os.path.join(output_dir, 'val_data.pt'))
        torch.save(test_data, os.path.join(output_dir, 'test_data.pt'))
        
        # Save indices for reference
        np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
        np.save(os.path.join(output_dir, 'val_indices.npy'), val_indices)
        np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
        
        # Return statistics
        stats = {
            'dataset_size': len(data_list),
            'num_nodes': node_features.shape[1],
            'num_edges': edge_index.shape[1],
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
    else:
        # Save data according to the specified split type
        if split_type == 'train':
            torch.save(data_list, os.path.join(output_dir, 'train_data.pt'))
        elif split_type == 'val':
            torch.save(data_list, os.path.join(output_dir, 'val_data.pt'))
        elif split_type == 'test':
            torch.save(data_list, os.path.join(output_dir, 'test_data.pt'))
        
        # Return statistics
        stats = {
            'dataset_size': len(data_list),
            'num_nodes': node_features.shape[1],
            'num_edges': edge_index.shape[1],
            f'{split_type}_size': len(data_list)
        }
    
    # Add feature selection and clustering information to statistics
    if use_advanced_feature_selection:
        stats['feature_selection'] = 'min_max_mutual_information'
        stats['num_selected_features'] = len(graph_constructor.selected_features) if graph_constructor.selected_features else 0
    
    if use_clustering:
        stats['clustering'] = 'kmeans_with_sbs'
        stats['num_clusters'] = num_clusters
        
        # Add cluster statistics
        if clustering_model and clustering_model.cluster_stats:
            stats['cluster_stats'] = clustering_model.cluster_stats
    
    # Save statistics
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Preprocessing completed. Data saved to " + output_dir)
    
    # Log statistics
    for key, value in stats.items():
        if key != 'cluster_stats':  # Skip detailed cluster stats in log
            logger.info(f"{key}: {value}")
    
    return stats

def main():
    """Main function."""
    # Set up logging
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Preprocess data
    stats = preprocess_data(
        data_file=args.data_file,
        mi_file=args.mi_file,
        output_dir=args.output_dir,
        mi_threshold=args.mi_threshold,
        use_advanced_feature_selection=args.use_advanced_feature_selection,
        top_features=args.top_features,
        use_clustering=args.use_clustering,
        num_clusters=args.num_clusters,
        split_type=args.split_type,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Print statistics
    print("\nPreprocessing Statistics:")
    for key, value in stats.items():
        if key != 'cluster_stats':  # Skip detailed cluster stats in output
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
