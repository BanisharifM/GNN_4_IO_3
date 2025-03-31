import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
import pickle
from datetime import datetime
from src.utils.feature_selection import AdvancedFeatureSelector
from src.utils.clustering import ClusteringManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IOCounterGraph:
    """
    Class for constructing graphs from I/O counter data.
    
    Each graph has:
    - Nodes: I/O counters
    - Edges: Based on mutual information between counters
    - Node features: Counter values
    - Target: Performance tag
    """
    
    def __init__(
        self, 
        mi_threshold: float = 0.3259,
        use_advanced_feature_selection: bool = False,
        n_features_to_select: int = 10,
        use_parallel: bool = True
    ):
        """
        Initialize the graph constructor.
        
        Args:
            mi_threshold (float): Threshold for mutual information to create an edge
            use_advanced_feature_selection (bool): Whether to use advanced feature selection
            n_features_to_select (int): Number of features to select if using advanced selection
            use_parallel (bool): Whether to use parallel processing for feature selection
        """
        self.mi_threshold = mi_threshold
        self.use_advanced_feature_selection = use_advanced_feature_selection
        self.n_features_to_select = n_features_to_select
        self.use_parallel = use_parallel
        self.counter_names = []
        self.counter_to_idx = {}  # Mapping from counter names to indices
        self.edge_index = None
        self.edge_attr = None
        self.selected_features = None
        self.feature_selector = None
    
    def load_mutual_information(self, mi_file_path: str) -> pd.DataFrame:
        """
        Load mutual information data from CSV file.
        
        Args:
            mi_file_path (str): Path to the mutual information CSV file
            
        Returns:
            pd.DataFrame: Mutual information data
        """
        logger.info(f"Loading mutual information from {mi_file_path}")
        mi_df = pd.read_csv(mi_file_path)
        return mi_df
    
    def construct_graph(self, mi_df: pd.DataFrame, data_df: Optional[pd.DataFrame] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph structure from mutual information data.
        
        Args:
            mi_df (pd.DataFrame): Mutual information data
            data_df (pd.DataFrame, optional): Data with counter values and target
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge index and edge attributes
        """
        if self.use_advanced_feature_selection and data_df is not None:
            logger.info("Using advanced feature selection")
            return self._construct_graph_with_advanced_selection(mi_df, data_df)
        else:
            logger.info(f"Constructing graph with MI threshold: {self.mi_threshold}")
            return self._construct_graph_with_threshold(mi_df)
    
    def _construct_graph_with_threshold(self, mi_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph using mutual information threshold.
        
        Args:
            mi_df (pd.DataFrame): Mutual information data
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge index and edge attributes
        """
        # Extract unique counter names
        counter1_list = mi_df['counter1'].unique().tolist()
        counter2_list = mi_df['counter2'].unique().tolist()
        self.counter_names = sorted(list(set(counter1_list + counter2_list)))
        
        # Create mapping from counter names to indices
        self.counter_to_idx = {name: idx for idx, name in enumerate(self.counter_names)}
        
        # Save counter mapping for later reference
        logger.info(f"Created mapping for {len(self.counter_names)} counters")
        
        # Create edges based on mutual information threshold
        edge_list = []
        edge_weights = []
        
        for _, row in mi_df.iterrows():
            counter1 = row['counter1']
            counter2 = row['counter2']
            mi_value = row['mutual_info']
            
            if mi_value >= self.mi_threshold:
                # Add edge in both directions (undirected graph)
                idx1 = self.counter_to_idx[counter1]
                idx2 = self.counter_to_idx[counter2]
                
                edge_list.append([idx1, idx2])
                edge_list.append([idx2, idx1])
                
                edge_weights.append(mi_value)
                edge_weights.append(mi_value)
        
        # Convert to PyTorch tensors
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        logger.info(f"Graph constructed with {len(self.counter_names)} nodes and {len(edge_weights)} edges")
        
        return self.edge_index, self.edge_attr
    
    def _construct_graph_with_advanced_selection(self, mi_df: pd.DataFrame, data_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph using advanced feature selection.
        
        Args:
            mi_df (pd.DataFrame): Mutual information data
            data_df (pd.DataFrame): Data with counter values and target
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge index and edge attributes
        """
        # Extract unique counter names
        counter1_list = mi_df['counter1'].unique().tolist()
        counter2_list = mi_df['counter2'].unique().tolist()
        all_counters = sorted(list(set(counter1_list + counter2_list)))
        
        # Create feature selector
        self.feature_selector = AdvancedFeatureSelector(
            n_features_to_select=self.n_features_to_select,
            random_state=42
        )
        
        # Select features that are present in the data
        available_counters = [counter for counter in all_counters if counter in data_df.columns]
        
        # Run feature selection
        if 'tag' in data_df.columns:
            selected_features = self.feature_selector.fit(
                data_df[available_counters + ['tag']], 
                'tag',
                self.use_parallel
            )
        else:
            logger.warning("No 'tag' column found in data, using all available counters")
            selected_features = available_counters
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Create mapping from counter names to indices
        self.counter_names = selected_features
        self.counter_to_idx = {name: idx for idx, name in enumerate(self.counter_names)}
        
        # Create edges based on mutual information between selected features
        edge_list = []
        edge_weights = []
        
        for _, row in mi_df.iterrows():
            counter1 = row['counter1']
            counter2 = row['counter2']
            mi_value = row['mutual_info']
            
            if counter1 in selected_features and counter2 in selected_features:
                # Add edge in both directions (undirected graph)
                idx1 = self.counter_to_idx[counter1]
                idx2 = self.counter_to_idx[counter2]
                
                edge_list.append([idx1, idx2])
                edge_list.append([idx2, idx1])
                
                edge_weights.append(mi_value)
                edge_weights.append(mi_value)
        
        # If no edges were created, create a fully connected graph
        if not edge_list:
            logger.warning("No edges created based on mutual information, creating fully connected graph")
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_weights.append(0.1)  # Default weight
                    edge_weights.append(0.1)
        
        # Convert to PyTorch tensors
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        logger.info(f"Graph constructed with {len(self.counter_names)} nodes and {len(edge_weights)} edges")
        
        return self.edge_index, self.edge_attr
    
    def get_node_features(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Extract node features from data.
        
        Args:
            data (pd.DataFrame): Data with counter values
            
        Returns:
            torch.Tensor: Node features for each sample
        """
        logger.info(f"Extracting node features for {len(data)} samples")
        
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
        Save counter mapping to file.
        
        Args:
            output_dir (str): Directory to save mapping
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save counter names and mapping
        with open(os.path.join(output_dir, 'counter_names.pkl'), 'wb') as f:
            pickle.dump(self.counter_names, f)
        
        with open(os.path.join(output_dir, 'counter_to_idx.pkl'), 'wb') as f:
            pickle.dump(self.counter_to_idx, f)
        
        # Also save as CSV for human readability
        mapping_df = pd.DataFrame({
            'counter_name': self.counter_names,
            'index': [self.counter_to_idx[name] for name in self.counter_names]
        })
        mapping_df.to_csv(os.path.join(output_dir, 'counter_mapping.csv'), index=False)
        
        # Save selected features if available
        if self.selected_features is not None:
            with open(os.path.join(output_dir, 'selected_features.pkl'), 'wb') as f:
                pickle.dump(self.selected_features, f)
            
            # Save as CSV for human readability
            pd.DataFrame({'feature': self.selected_features}).to_csv(
                os.path.join(output_dir, 'selected_features.csv'), 
                index=False
            )
        
        # Save feature importances if available
        if self.feature_selector is not None:
            importances = self.feature_selector.get_feature_importances()
            if importances:
                importances_df = pd.DataFrame({
                    'feature': list(importances.keys()),
                    'importance': list(importances.values())
                }).sort_values('importance', ascending=False)
                
                importances_df.to_csv(
                    os.path.join(output_dir, 'feature_importances.csv'), 
                    index=False
                )
        
        logger.info(f"Counter mapping saved to {output_dir}")
    
    def load_counter_mapping(self, input_dir: str):
        """
        Load counter mapping from file.
        
        Args:
            input_dir (str): Directory with saved mapping
        """
        # Load counter names
        with open(os.path.join(input_dir, 'counter_names.pkl'), 'rb') as f:
            self.counter_names = pickle.load(f)
        
        # Load counter mapping
        with open(os.path.join(input_dir, 'counter_to_idx.pkl'), 'rb') as f:
            self.counter_to_idx = pickle.load(f)
        
        # Load selected features if available
        selected_features_path = os.path.join(input_dir, 'selected_features.pkl')
        if os.path.exists(selected_features_path):
            with open(selected_features_path, 'rb') as f:
                self.selected_features = pickle.load(f)
        
        logger.info(f"Counter mapping loaded from {input_dir}")

class IOGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for I/O counter graphs.
    """
    
    def __init__(
        self, 
        data_file: str, 
        mi_file: str, 
        mi_threshold: float = 0.3259,
        use_advanced_feature_selection: bool = False,
        n_features_to_select: int = 10,
        use_parallel: bool = True,
        use_clustering: bool = False,
        n_clusters: int = 4,
        transform=None, 
        pre_transform=None,
        root: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_file (str): Path to the data CSV file
            mi_file (str): Path to the mutual information CSV file
            mi_threshold (float): Threshold for mutual information
            use_advanced_feature_selection (bool): Whether to use advanced feature selection
            n_features_to_select (int): Number of features to select if using advanced selection
            use_parallel (bool): Whether to use parallel processing for feature selection
            use_clustering (bool): Whether to use clustering
            n_clusters (int): Number of clusters if using clustering
            transform: PyTorch Geometric transform
            pre_transform: PyTorch Geometric pre-transform
            root: Root directory for dataset
        """
        super(IOGraphDataset, self).__init__(root, transform, pre_transform)
        self.data_file = data_file
        self.mi_file = mi_file
        self.mi_threshold = mi_threshold
        self.use_advanced_feature_selection = use_advanced_feature_selection
        self.n_features_to_select = n_features_to_select
        self.use_parallel = use_parallel
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.data_list = None
        self.cluster_labels = None
        self.cluster_datasets = None
        
        # Process data
        self.process()
    
    def process(self):
        """
        Process the raw data into PyTorch Geometric Data objects.
        """
        logger.info("Processing data into PyTorch Geometric format")
        
        # Load data
        data_df = pd.read_csv(self.data_file)
        
        # Initialize graph constructor
        graph_constructor = IOCounterGraph(
            mi_threshold=self.mi_threshold,
            use_advanced_feature_selection=self.use_advanced_feature_selection,
            n_features_to_select=self.n_features_to_select,
            use_parallel=self.use_parallel
        )
        
        # Load mutual information and construct graph
        mi_df = graph_constructor.load_mutual_information(self.mi_file)
        edge_index, edge_attr = graph_constructor.construct_graph(mi_df, data_df)
        
        # Extract node features
        node_features = graph_constructor.get_node_features(data_df)
        
        # Extract targets
        targets = torch.tensor(data_df['tag'].values, dtype=torch.float)
        
        # Create data list
        self.data_list = []
        for i in range(len(data_df)):
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features[i].view(-1, 1),  # [num_nodes, 1]
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=targets[i].view(-1)  # [1]
            )
            
            self.data_list.append(data)
        
        logger.info(f"Created dataset with {len(self.data_list)} samples")
        
        # Save counter mapping
        if self.root is not None:
            graph_constructor.save_counter_mapping(self.root)
        
        # Apply clustering if requested
        if self.use_clustering:
            self._apply_clustering(data_df, graph_constructor.counter_names)
    
    def _apply_clustering(self, data_df: pd.DataFrame, features: List[str]):
        """
        Apply clustering to the dataset.
        
        Args:
            data_df (pd.DataFrame): Input data
            features (List[str]): Features to use for clustering
        """
        logger.info(f"Applying clustering with {self.n_clusters} clusters")
        
        # Create clustering directory
        clustering_dir = os.path.join(self.root, 'clustering')
        os.makedirs(clustering_dir, exist_ok=True)
        
        # Initialize clustering manager
        clustering_manager = ClusteringManager(
            n_clusters=self.n_clusters,
            random_state=42,
            output_dir=clustering_dir
        )
        
        # Filter features that are present in the data
        available_features = [f for f in features if f in data_df.columns]
        
        # Fit clustering
        self.cluster_labels = clustering_manager.fit(data_df, available_features)
        
        # Create separate datasets for each cluster
        self.cluster_datasets = {}
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            self.cluster_datasets[cluster_id] = [self.data_list[i] for i in cluster_indices]
            
            logger.info(f"Cluster {cluster_id}: {len(self.cluster_datasets[cluster_id])} samples")
        
        # Save cluster labels
        cluster_labels_df = pd.DataFrame({
            'index': range(len(self.cluster_labels)),
            'cluster': self.cluster_labels
        })
        cluster_labels_df.to_csv(os.path.join(clustering_dir, 'cluster_labels.csv'), index=False)
    
    def len(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data_list)
    
    def get(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        return self.data_list[idx]
    
    def get_cluster_dataset(self, cluster_id):
        """
        Get dataset for a specific cluster.
        
        Args:
            cluster_id (int): Cluster ID
            
        Returns:
            List[Data]: List of PyTorch Geometric Data objects for the cluster
        """
        if not self.use_clustering or self.cluster_datasets is None:
            raise ValueError("Clustering not applied to this dataset")
        
        if cluster_id not in self.cluster_datasets:
            raise ValueError(f"Invalid cluster ID: {cluster_id}")
        
        return self.cluster_datasets[cluster_id]

def create_dataset(
    data_file: str, 
    mi_file: str, 
    mi_threshold: float = 0.3259,
    use_advanced_feature_selection: bool = False,
    n_features_to_select: int = 10,
    use_parallel: bool = True,
    use_clustering: bool = False,
    n_clusters: int = 4,
    output_dir: Optional[str] = None
) -> IOGraphDataset:
    """
    Create a PyTorch Geometric dataset from raw data.
    
    Args:
        data_file (str): Path to the data CSV file
        mi_file (str): Path to the mutual information CSV file
        mi_threshold (float): Threshold for mutual information
        use_advanced_feature_selection (bool): Whether to use advanced feature selection
        n_features_to_select (int): Number of features to select if using advanced selection
        use_parallel (bool): Whether to use parallel processing for feature selection
        use_clustering (bool): Whether to use clustering
        n_clusters (int): Number of clusters if using clustering
        output_dir (str, optional): Directory to save processed data
        
    Returns:
        IOGraphDataset: PyTorch Geometric dataset
    """
    return IOGraphDataset(
        data_file=data_file,
        mi_file=mi_file,
        mi_threshold=mi_threshold,
        use_advanced_feature_selection=use_advanced_feature_selection,
        n_features_to_select=n_features_to_select,
        use_parallel=use_parallel,
        use_clustering=use_clustering,
        n_clusters=n_clusters,
        root=output_dir
    )
