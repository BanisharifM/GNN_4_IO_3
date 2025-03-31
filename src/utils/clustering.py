import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ClusteringManager:
    """
    Clustering manager for grouping jobs with similar I/O characteristics.
    
    Implements clustering approach as described in the HiPC21 paper.
    """
    
    def __init__(
        self, 
        n_clusters: int = 4, 
        random_state: int = 42,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the clustering manager.
        
        Args:
            n_clusters (int): Number of clusters for KMeans
            random_state (int): Random state for reproducibility
            output_dir (str, optional): Directory to save clustering results
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.output_dir = output_dir
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_stats = {}
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def fit(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Fit KMeans clustering on the data.
        
        Args:
            data (pd.DataFrame): Input data
            features (List[str]): Features to use for clustering
            
        Returns:
            np.ndarray: Cluster labels
        """
        logger.info(f"Fitting KMeans clustering with {self.n_clusters} clusters on {len(features)} features")
        
        # Extract features
        X = data[features].values
        
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(data, features, labels)
        
        # Save clustering model
        if self.output_dir is not None:
            self._save_clustering_model()
            self._save_cluster_stats()
            self._visualize_clusters(data, features, labels)
        
        logger.info(f"KMeans clustering completed with {self.n_clusters} clusters")
        
        return labels
    
    def predict(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            data (pd.DataFrame): Input data
            features (List[str]): Features to use for clustering
            
        Returns:
            np.ndarray: Cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Clustering model not fitted yet")
        
        # Extract features
        X = data[features].values
        
        # Standardize data
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        labels = self.kmeans.predict(X_scaled)
        
        return labels
    
    def _calculate_cluster_stats(self, data: pd.DataFrame, features: List[str], labels: np.ndarray) -> None:
        """
        Calculate statistics for each cluster.
        
        Args:
            data (pd.DataFrame): Input data
            features (List[str]): Features used for clustering
            labels (np.ndarray): Cluster labels
        """
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Calculate statistics for each cluster
        for cluster_id in range(self.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            # Basic statistics
            self.cluster_stats[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'feature_means': {},
                'feature_stds': {}
            }
            
            # Feature statistics
            for feature in features:
                self.cluster_stats[cluster_id]['feature_means'][feature] = cluster_data[feature].mean()
                self.cluster_stats[cluster_id]['feature_stds'][feature] = cluster_data[feature].std()
            
            # If 'tag' column exists, calculate performance statistics
            if 'tag' in data.columns:
                self.cluster_stats[cluster_id]['tag_mean'] = cluster_data['tag'].mean()
                self.cluster_stats[cluster_id]['tag_std'] = cluster_data['tag'].std()
                self.cluster_stats[cluster_id]['tag_min'] = cluster_data['tag'].min()
                self.cluster_stats[cluster_id]['tag_max'] = cluster_data['tag'].max()
    
    def _save_clustering_model(self) -> None:
        """
        Save clustering model to file.
        """
        model_path = os.path.join(self.output_dir, 'kmeans_model.pkl')
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Clustering model saved to {model_path}")
    
    def _save_cluster_stats(self) -> None:
        """
        Save cluster statistics to file.
        """
        stats_path = os.path.join(self.output_dir, 'cluster_stats.json')
        
        # Convert numpy types to Python types for JSON serialization
        stats_json = {}
        for cluster_id, stats in self.cluster_stats.items():
            stats_json[str(cluster_id)] = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    stats_json[str(cluster_id)][key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (np.integer, np.floating)):
                            stats_json[str(cluster_id)][key][subkey] = float(subvalue)
                        else:
                            stats_json[str(cluster_id)][key][subkey] = subvalue
                elif isinstance(value, (np.integer, np.floating)):
                    stats_json[str(cluster_id)][key] = float(value)
                else:
                    stats_json[str(cluster_id)][key] = value
        
        # Save as JSON
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=4)
        
        logger.info(f"Cluster statistics saved to {stats_path}")
        
        # Also save as CSV for easier analysis
        stats_df = pd.DataFrame()
        for cluster_id, stats in self.cluster_stats.items():
            row = {'cluster_id': cluster_id, 'size': stats['size'], 'percentage': stats['percentage']}
            
            # Add feature means
            for feature, mean in stats['feature_means'].items():
                row[f'{feature}_mean'] = mean
            
            # Add performance statistics if available
            if 'tag_mean' in stats:
                row['tag_mean'] = stats['tag_mean']
                row['tag_std'] = stats['tag_std']
                row['tag_min'] = stats['tag_min']
                row['tag_max'] = stats['tag_max']
            
            stats_df = pd.concat([stats_df, pd.DataFrame([row])], ignore_index=True)
        
        stats_df.to_csv(os.path.join(self.output_dir, 'cluster_stats.csv'), index=False)
    
    def _visualize_clusters(self, data: pd.DataFrame, features: List[str], labels: np.ndarray) -> None:
        """
        Visualize clustering results.
        
        Args:
            data (pd.DataFrame): Input data
            features (List[str]): Features used for clustering
            labels (np.ndarray): Cluster labels
        """
        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # 1. Cluster sizes
        plt.figure(figsize=(10, 6))
        cluster_sizes = [self.cluster_stats[i]['size'] for i in range(self.n_clusters)]
        plt.bar(range(self.n_clusters), cluster_sizes)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Jobs')
        plt.title('Cluster Sizes')
        plt.xticks(range(self.n_clusters))
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(vis_dir, 'cluster_sizes.png'), dpi=300)
        plt.close()
        
        # 2. Feature distributions by cluster
        for feature in features:
            plt.figure(figsize=(12, 8))
            for cluster_id in range(self.n_clusters):
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                sns.kdeplot(cluster_data[feature], label=f'Cluster {cluster_id}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.title(f'Distribution of {feature} by Cluster')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, f'cluster_dist_{feature}.png'), dpi=300)
            plt.close()
        
        # 3. Performance (tag) by cluster if available
        if 'tag' in data.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='cluster', y='tag', data=data_with_clusters)
            plt.xlabel('Cluster ID')
            plt.ylabel('Performance Tag')
            plt.title('Performance Tag by Cluster')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'cluster_performance.png'), dpi=300)
            plt.close()
        
        # 4. 2D visualization of clusters (first two features)
        if len(features) >= 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                data_with_clusters[features[0]], 
                data_with_clusters[features[1]], 
                c=labels, 
                cmap='viridis', 
                alpha=0.5
            )
            plt.colorbar(scatter, label='Cluster ID')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title('2D Visualization of Clusters')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'cluster_2d.png'), dpi=300)
            plt.close()
        
        # 5. Cluster centroids
        centroids = self.kmeans.cluster_centers_
        centroids_df = pd.DataFrame(centroids, columns=[f'{feature}_scaled' for feature in features])
        
        # Inverse transform to get original scale
        centroids_original = self.scaler.inverse_transform(centroids)
        centroids_df_original = pd.DataFrame(centroids_original, columns=features)
        
        # Save centroids
        centroids_df.to_csv(os.path.join(self.output_dir, 'cluster_centroids_scaled.csv'), index=False)
        centroids_df_original.to_csv(os.path.join(self.output_dir, 'cluster_centroids_original.csv'), index=False)
        
        logger.info(f"Clustering visualizations saved to {vis_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'ClusteringManager':
        """
        Load clustering model from file.
        
        Args:
            model_dir (str): Directory with saved model
            
        Returns:
            ClusteringManager: Loaded clustering manager
        """
        model_path = os.path.join(model_dir, 'kmeans_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}")
        
        # Load KMeans model
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Create clustering manager
        manager = cls(n_clusters=kmeans.n_clusters, output_dir=model_dir)
        manager.kmeans = kmeans
        manager.scaler = scaler
        
        # Load cluster statistics if available
        stats_path = os.path.join(model_dir, 'cluster_stats.json')
        if os.path.exists(stats_path):
            import json
            with open(stats_path, 'r') as f:
                stats_json = json.load(f)
            
            # Convert string keys to integers
            manager.cluster_stats = {int(k): v for k, v in stats_json.items()}
        
        logger.info(f"Clustering model loaded from {model_dir}")
        
        return manager
