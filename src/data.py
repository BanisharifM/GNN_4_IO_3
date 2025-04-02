import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional

class IOCounterGraph:
    """
    Class for constructing graphs from I/O counter data.
    """
    def __init__(self, mi_threshold: float = 0.3259):
        """
        Initialize the graph constructor.
        
        Args:
            mi_threshold: Threshold for mutual information to create an edge
        """
        self.mi_threshold = mi_threshold
        self.counter_names = []
        self.counter_to_idx = {}
        self.edge_index = None
        self.edge_attr = None
        self.logger = logging.getLogger(__name__)
        
    def load_mutual_information(self, mi_file_path: str) -> pd.DataFrame:
        """
        Load mutual information data from CSV file.
        
        Args:
            mi_file_path: Path to mutual information CSV file
            
        Returns:
            DataFrame containing mutual information values
        """
        self.logger.info(f"Loading mutual information from {mi_file_path}")
        
        # Read the CSV file directly - it's already in matrix format
        mi_df = pd.read_csv(mi_file_path, index_col=0)
        
        return mi_df
        
    def construct_graph(self, mi_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct a graph from mutual information data.
        
        Args:
            mi_df: DataFrame containing mutual information values
            
        Returns:
            Tuple of (edge_index, edge_attr) tensors
        """
        self.logger.info(f"Constructing graph with MI threshold: {self.mi_threshold}")
        
        # Get counter names from the DataFrame columns (excluding the index column)
        self.counter_names = [col for col in mi_df.columns if col != 'tag' and col != 'nprocs']
        
        # Create mapping from counter name to index
        self.counter_to_idx = {name: idx for idx, name in enumerate(self.counter_names)}
        
        # Construct graph using threshold
        return self._construct_graph_with_threshold(mi_df)
    
    def _construct_graph_with_threshold(self, mi_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct a graph using a threshold on mutual information.
        
        Args:
            mi_df: DataFrame containing mutual information values
            
        Returns:
            Tuple of (edge_index, edge_attr) tensors
        """
        edge_list = []
        edge_weights = []
        
        # Iterate through all pairs of counters
        for i, counter1 in enumerate(self.counter_names):
            for j, counter2 in enumerate(self.counter_names):
                if i >= j:  # Skip self-loops and duplicates
                    continue
                
                # Get mutual information value - handle potential missing values
                try:
                    mi_value = float(mi_df.loc[counter1, counter2])
                    
                    # Add edge if mutual information is above threshold
                    if mi_value >= self.mi_threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])  # Add reverse edge
                        edge_weights.append(mi_value)
                        edge_weights.append(mi_value)
                except (KeyError, ValueError):
                    # Skip if value is missing or not a number
                    continue
        
        # Convert to PyTorch tensors
        if not edge_list:  # Handle case with no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
        self.logger.info(f"Created graph with {len(self.counter_names)} nodes and {len(edge_weights)//2} edges")
        
        return edge_index, edge_attr
    
    def get_node_features(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Extract node features from data.
        
        Args:
            data: DataFrame containing counter values
            
        Returns:
            Tensor of node features
        """
        features_list = []
        
        for _, row in data.iterrows():
            # Extract counter values for this sample
            node_features = []
            for counter in self.counter_names:
                if counter in row:
                    node_features.append(float(row[counter]))
                else:
                    node_features.append(0.0)
            features_list.append(node_features)
        
        return torch.tensor(features_list, dtype=torch.float)
    
    def save_counter_mapping(self, output_dir: str):
        """
        Save counter to index mapping to a file.
        
        Args:
            output_dir: Directory to save the mapping
        """
        os.makedirs(output_dir, exist_ok=True)
        mapping_file = os.path.join(output_dir, "counter_mapping.csv")
        
        mapping_df = pd.DataFrame({
            "counter_name": list(self.counter_to_idx.keys()),
            "index": list(self.counter_to_idx.values())
        })
        
        mapping_df.to_csv(mapping_file, index=False)
        self.logger.info(f"Saved counter mapping to {mapping_file}")
    
    def load_counter_mapping(self, mapping_file: str):
        """
        Load counter to index mapping from a file.
        
        Args:
            mapping_file: Path to mapping file
        """
        mapping_df = pd.read_csv(mapping_file)
        self.counter_names = mapping_df["counter_name"].tolist()
        self.counter_to_idx = dict(zip(mapping_df["counter_name"], mapping_df["index"]))
        self.logger.info(f"Loaded counter mapping from {mapping_file}")
        
    def export_graph_structure(self, output_dir: str):
        """
        Export the graph structure to JSON, CSV, and PyTorch (.pt) formats.
        
        Args:
            output_dir: Directory to save the graph structure
        """
        os.makedirs(output_dir, exist_ok=True)
        graph_file = os.path.join(output_dir, "graph_structure.json")
        
        # Create a dictionary representation of the graph
        graph_dict = {
            "nodes": self.counter_names,
            "edges": []
        }
        
        # Add edges with their weights
        if self.edge_index is not None and self.edge_index.shape[1] > 0:
            for i in range(0, self.edge_index.shape[1], 2):  # Only process one direction
                source_idx = self.edge_index[0, i].item()
                target_idx = self.edge_index[1, i].item()
                weight = self.edge_attr[i, 0].item()
                
                source_name = self.counter_names[source_idx]
                target_name = self.counter_names[target_idx]
                
                graph_dict["edges"].append({
                    "source": source_name,
                    "target": target_name,
                    "weight": weight
                })
        
        # Save to JSON file
        with open(graph_file, 'w') as f:
            json.dump(graph_dict, f, indent=2)
        
        self.logger.info(f"Exported graph structure to {graph_file}")
        
        # Also export as CSV for easier analysis
        edges_df = pd.DataFrame(graph_dict["edges"])
        if not edges_df.empty:
            csv_file = os.path.join(output_dir, "graph_edges.csv")
            edges_df.to_csv(csv_file, index=False)
            self.logger.info(f"Exported graph edges to {csv_file}")
        
        # Export node list
        nodes_df = pd.DataFrame({"counter_name": self.counter_names})
        nodes_csv = os.path.join(output_dir, "graph_nodes.csv")
        nodes_df.to_csv(nodes_csv, index=False)
        self.logger.info(f"Exported graph nodes to {nodes_csv}")
        
        # Save edge_index and edge_attr as PyTorch tensors with the exact filenames expected by 02_train_model.py
        if self.edge_index is not None and self.edge_attr is not None:
            # Save edge_index tensor
            edge_index_file = os.path.join(output_dir, "edge_index.pt")
            torch.save(self.edge_index, edge_index_file)
            self.logger.info(f"Exported graph edges tensor to {edge_index_file}")
            
            # Save edge_attr tensor
            edge_attr_file = os.path.join(output_dir, "edge_attr.pt")
            torch.save(self.edge_attr, edge_attr_file)
            self.logger.info(f"Exported graph edge attributes tensor to {edge_attr_file}")


class IOGraphDataset:
    """
    Dataset class for I/O counter graphs.
    """
    def __init__(self, 
                 data_file: str, 
                 mi_file: str, 
                 mi_threshold: float = 0.3259,
                 use_advanced_feature_selection: bool = False,
                 use_clustering: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to data CSV file
            mi_file: Path to mutual information CSV file
            mi_threshold: Threshold for mutual information to create an edge
            use_advanced_feature_selection: Whether to use advanced feature selection
            use_clustering: Whether to use clustering
        """
        self.data_file = data_file
        self.mi_file = mi_file
        self.mi_threshold = mi_threshold
        self.use_advanced_feature_selection = use_advanced_feature_selection
        self.use_clustering = use_clustering
        self.logger = logging.getLogger(__name__)
        
    def process(self) -> List[Data]:
        """
        Process the dataset.
        
        Returns:
            List of PyTorch Geometric Data objects
        """
        # Load data
        self.logger.info(f"Loading data from {self.data_file}")
        data_df = pd.read_csv(self.data_file)
        
        # Extract targets
        targets = torch.tensor(data_df['tag'].values, dtype=torch.float)
        
        # Construct graph
        graph_constructor = IOCounterGraph(mi_threshold=self.mi_threshold)
        mi_df = graph_constructor.load_mutual_information(self.mi_file)
        edge_index, edge_attr = graph_constructor.construct_graph(mi_df)
        
        # Get node features
        node_features = graph_constructor.get_node_features(data_df)
        
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
            data_list.append(data)
        
        return data_list, graph_constructor
