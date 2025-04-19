import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

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
        self.selected_features = None
        
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
    
    def min_max_mutual_information_selection(self, mi_df: pd.DataFrame, target_col: str = 'tag', top_n: int = 10) -> List[str]:
        """
        Implement Min-max mutual information feature selection from HiPC21 paper.
        
        Args:
            mi_df: DataFrame containing mutual information values
            target_col: Target column name
            top_n: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Performing Min-max mutual information feature selection to select top {top_n} features")
        
        # Get all counter names excluding the target column
        all_counters = [col for col in mi_df.columns if col != target_col and col != 'nprocs']
        
        # Calculate correlation with target for all features
        target_correlations = {}
        for counter in all_counters:
            if target_col in mi_df.index and counter in mi_df.columns:
                target_correlations[counter] = abs(float(mi_df.loc[target_col, counter]))
            else:
                target_correlations[counter] = 0.0
        
        # Sort features by correlation with target
        sorted_by_target = sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Select the first feature (most correlated with target)
        selected_features = [sorted_by_target[0][0]]
        
        # Iteratively select remaining features
        while len(selected_features) < min(top_n, len(all_counters)):
            # Calculate correlation with already selected features
            feature_correlations = {}
            
            for counter in all_counters:
                if counter in selected_features:
                    continue
                
                # Calculate average correlation with already selected features
                avg_correlation = 0.0
                for selected in selected_features:
                    if selected in mi_df.index and counter in mi_df.columns:
                        avg_correlation += abs(float(mi_df.loc[selected, counter]))
                    elif counter in mi_df.index and selected in mi_df.columns:
                        avg_correlation += abs(float(mi_df.loc[counter, selected]))
                
                avg_correlation /= len(selected_features)
                feature_correlations[counter] = avg_correlation
            
            # Sort by correlation with selected features (ascending)
            sorted_by_correlation = sorted(feature_correlations.items(), key=lambda x: x[1])
            
            # Take top 10 least correlated features
            candidates = sorted_by_correlation[:10]
            
            # From these candidates, select the one with highest correlation to target
            best_candidate = max(candidates, key=lambda x: target_correlations[x[0]])
            
            # Add to selected features
            selected_features.append(best_candidate[0])
        
        self.logger.info(f"Selected features: {selected_features}")
        self.selected_features = selected_features
        return selected_features
        
    def construct_graph(self, mi_df: pd.DataFrame, use_advanced_feature_selection: bool = False, target_col: str = 'tag', top_n: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct a graph from mutual information data.
        
        Args:
            mi_df: DataFrame containing mutual information values
            use_advanced_feature_selection: Whether to use advanced feature selection
            target_col: Target column name for feature selection
            top_n: Number of top features to select
            
        Returns:
            Tuple of (edge_index, edge_attr) tensors
        """
        self.logger.info(f"Constructing graph with MI threshold: {self.mi_threshold}")
        
        # Get counter names from the DataFrame columns (excluding the index column)
        all_counters = [col for col in mi_df.columns if col != target_col and col != 'nprocs']
        
        # Apply feature selection if requested
        if use_advanced_feature_selection:
            self.logger.info("Using advanced feature selection")
            selected_features = self.min_max_mutual_information_selection(mi_df, target_col, top_n)
            self.counter_names = selected_features
            
            # Filter mutual information matrix to only include selected features
            filtered_mi_df = mi_df.loc[selected_features, selected_features]
            
            # Create mapping from counter name to index
            self.counter_to_idx = {name: idx for idx, name in enumerate(selected_features)}
            
            # Construct graph using threshold and filtered MI matrix
            return self._construct_graph_with_threshold(filtered_mi_df)
        else:
            # Use all counters
            self.counter_names = all_counters
            
            # Create mapping from counter name to index
            self.counter_to_idx = {name: idx for idx, name in enumerate(all_counters)}
            
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
        Export the graph structure to JSON format.
        
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
        
        # Save edge_index and edge_attr as PyTorch tensors
        if self.edge_index is not None:
            torch.save(self.edge_index, os.path.join(output_dir, 'edge_index.pt'))
            self.logger.info(f"Saved edge_index tensor to {os.path.join(output_dir, 'edge_index.pt')}")
        
        if self.edge_attr is not None:
            torch.save(self.edge_attr, os.path.join(output_dir, 'edge_attr.pt'))
            self.logger.info(f"Saved edge_attr tensor to {os.path.join(output_dir, 'edge_attr.pt')}")


class ClusteringModel:
    """
    Class for clustering I/O data based on the HiPC21 paper approach.
    """
    def __init__(self, n_clusters: int = 4):
        """
        Initialize the clustering model.
        
        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.selected_features = None
        self.cluster_stats = None
        
    def sequential_backward_selection(self, data: pd.DataFrame, initial_features: List[str], min_features: int = 3) -> List[str]:
        """
        Implement Sequential Backward Selection (SBS) for feature selection in clustering.
        
        Args:
            data: DataFrame containing feature values
            initial_features: Initial set of features to start with
            min_features: Minimum number of features to keep
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Performing Sequential Backward Selection starting with {len(initial_features)} features")
        
        current_features = initial_features.copy()
        best_score = -float('inf')
        best_features = current_features.copy()
        
        while len(current_features) > min_features:
            feature_subsets = []
            
            # Create feature subsets by removing one feature at a time
            for i in range(len(current_features)):
                subset = current_features.copy()
                removed = subset.pop(i)
                feature_subsets.append((subset, removed))
            
            # Evaluate each subset
            subset_scores = []
            for subset, removed in feature_subsets:
                # Extract features for clustering
                X = data[subset].values
                X = self.scaler.fit_transform(X)
                
                # Perform clustering
                self.kmeans.fit(X)
                labels = self.kmeans.labels_
                
                # Calculate evaluation metrics
                if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
                    silhouette = silhouette_score(X, labels)
                    dbi = davies_bouldin_score(X, labels)
                    
                    # Combined score (higher is better)
                    combined_score = silhouette / dbi
                else:
                    combined_score = -float('inf')
                
                subset_scores.append((subset, removed, combined_score))
            
            # Find the best subset
            best_subset = max(subset_scores, key=lambda x: x[2])
            
            # Update current features
            current_features = best_subset[0]
            
            # Update best features if score improved
            if best_subset[2] > best_score:
                best_score = best_subset[2]
                best_features = current_features.copy()
            
            self.logger.info(f"Removed feature '{best_subset[1]}', new score: {best_subset[2]:.4f}, remaining features: {len(current_features)}")
        
        self.logger.info(f"Selected features after SBS: {best_features}")
        self.selected_features = best_features
        return best_features
    
    def fit(self, data: pd.DataFrame, target_col: str = 'tag', use_sbs: bool = True, initial_features: List[str] = None) -> np.ndarray:
        """
        Fit the clustering model to the data.
        
        Args:
            data: DataFrame containing feature values
            target_col: Target column name
            use_sbs: Whether to use Sequential Backward Selection
            initial_features: Initial set of features to use (if None, use all features except target)
            
        Returns:
            Cluster assignments for each data point
        """
        self.logger.info(f"Fitting clustering model with {self.n_clusters} clusters")
        
        # Determine features to use
        if initial_features is None:
            initial_features = [col for col in data.columns if col != target_col and col != 'nprocs']
        
        # Apply SBS if requested
        if use_sbs:
            selected_features = self.sequential_backward_selection(data, initial_features)
        else:
            selected_features = initial_features
            self.selected_features = selected_features
        
        # Extract features for clustering
        X = data[selected_features].values
        X = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        
        # Calculate cluster statistics
        self.calculate_cluster_statistics(data, labels, target_col)
        
        return labels
    
    def calculate_cluster_statistics(self, data: pd.DataFrame, labels: np.ndarray, target_col: str = 'tag'):
        """
        Calculate statistics for each cluster.
        
        Args:
            data: DataFrame containing feature values
            labels: Cluster assignments
            target_col: Target column name
        """
        self.logger.info("Calculating cluster statistics")
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Calculate statistics for each cluster
        cluster_stats = {}
        for cluster_id in range(self.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            # Calculate statistics for target variable
            target_mean = cluster_data[target_col].mean()
            target_std = cluster_data[target_col].std()
            target_count = len(cluster_data)
            
            # Store statistics
            cluster_stats[cluster_id] = {
                'count': target_count,
                'target_mean': target_mean,
                'target_std': target_std
            }
            
            self.logger.info(f"Cluster {cluster_id}: {target_count} samples, target mean: {target_mean:.4f}, target std: {target_std:.4f}")
        
        self.cluster_stats = cluster_stats
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster assignments for new data.
        
        Args:
            data: DataFrame containing feature values
            
        Returns:
            Cluster assignments for each data point
        """
        if self.selected_features is None:
            raise ValueError("Model has not been fitted yet")
        
        # Extract features for clustering
        X = data[self.selected_features].values
        X = self.scaler.transform(X)
        
        # Predict clusters
        labels = self.kmeans.predict(X)
        
        return labels
    
    def export_cluster_info(self, output_dir: str, data: pd.DataFrame = None, labels: np.ndarray = None):
        """
        Export clustering information to files.
        
        Args:
            output_dir: Directory to save the clustering information
            data: DataFrame containing feature values (optional)
            labels: Cluster assignments (optional)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export selected features
        if self.selected_features is not None:
            features_file = os.path.join(output_dir, "cluster_features.json")
            with open(features_file, 'w') as f:
                json.dump(self.selected_features, f, indent=2)
            self.logger.info(f"Exported cluster features to {features_file}")
        
        # Export cluster statistics
        if self.cluster_stats is not None:
            stats_file = os.path.join(output_dir, "cluster_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.cluster_stats, f, indent=2)
            self.logger.info(f"Exported cluster statistics to {stats_file}")
        
        # Export cluster centers
        if hasattr(self.kmeans, 'cluster_centers_'):
            centers_file = os.path.join(output_dir, "cluster_centers.npy")
            np.save(centers_file, self.kmeans.cluster_centers_)
            self.logger.info(f"Exported cluster centers to {centers_file}")
        
        # Export data with cluster labels
        if data is not None and labels is not None:
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = labels
            
            # Save to CSV
            csv_file = os.path.join(output_dir, "data_with_clusters.csv")
            data_with_clusters.to_csv(csv_file, index=False)
            self.logger.info(f"Exported data with cluster labels to {csv_file}")


class IOGraphDataset:
    """
    Dataset class for I/O counter graphs.
    """
    def __init__(self, 
                 data_file: str, 
                 mi_file: str, 
                 mi_threshold: float = 0.3259,
                 use_advanced_feature_selection: bool = False,
                 use_clustering: bool = False,
                 top_features: int = 10,
                 n_clusters: int = 4):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to data CSV file
            mi_file: Path to mutual information CSV file
            mi_threshold: Threshold for mutual information to create an edge
            use_advanced_feature_selection: Whether to use advanced feature selection
            use_clustering: Whether to use clustering
            top_features: Number of top features to select for advanced feature selection
            n_clusters: Number of clusters for clustering
        """
        self.data_file = data_file
        self.mi_file = mi_file
        self.mi_threshold = mi_threshold
        self.use_advanced_feature_selection = use_advanced_feature_selection
        self.use_clustering = use_clustering
        self.top_features = top_features
        self.n_clusters = n_clusters
        self.logger = logging.getLogger(__name__)
        
    def process(self, output_dir: str = None) -> Tuple[List[Data], IOCounterGraph, Optional[ClusteringModel]]:
        """
        Process the dataset.
        
        Args:
            output_dir: Directory to save intermediate results (optional)
            
        Returns:
            Tuple of (data_list, graph_constructor, clustering_model)
        """
        # Load data
        self.logger.info(f"Loading data from {self.data_file}")
        data_df = pd.read_csv(self.data_file)
        
        # Extract targets
        targets = torch.tensor(data_df['tag'].values, dtype=torch.float)
        
        # Construct graph
        graph_constructor = IOCounterGraph(mi_threshold=self.mi_threshold)
        mi_df = graph_constructor.load_mutual_information(self.mi_file)
        
        # Apply advanced feature selection if requested
        edge_index, edge_attr = graph_constructor.construct_graph(
            mi_df, 
            use_advanced_feature_selection=self.use_advanced_feature_selection,
            target_col='tag',
            top_n=self.top_features
        )
        
        # Apply clustering if requested
        clustering_model = None
        if self.use_clustering:
            self.logger.info("Applying clustering to data")
            clustering_model = ClusteringModel(n_clusters=self.n_clusters)
            
            # Use selected features from feature selection if available
            initial_features = graph_constructor.selected_features if graph_constructor.selected_features else None
            
            # Fit clustering model
            cluster_labels = clustering_model.fit(data_df, target_col='tag', initial_features=initial_features)
            
            # Add cluster labels to data
            data_df['cluster'] = cluster_labels
            
            # Export clustering information if output directory is provided
            if output_dir:
                clustering_model.export_cluster_info(output_dir, data_df, cluster_labels)
        
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
            
            # Add cluster information if available
            if self.use_clustering:
                data.cluster = torch.tensor([data_df.iloc[i]['cluster']], dtype=torch.long)
            
            data_list.append(data)
        
        return data_list, graph_constructor, clustering_model
