import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap
from typing import Optional, List, Tuple, Dict, Union
import logging
from omegaconf import DictConfig
from ..models.gnn import GNNModel, GNNRegressor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GNNExplainer:
    """
    Class for explaining GNN predictions using SHAP values.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize the GNN explainer.
        
        Args:
            model (torch.nn.Module): Trained GNN model
            device (torch.device): Device to run the model on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Initialized GNN explainer with model on {device}")
    
    def _prepare_background_data(self, data_list: List, num_samples: int = 100) -> List:
        """
        Prepare background data for SHAP analysis.
        
        Args:
            data_list (List): List of PyTorch Geometric Data objects
            num_samples (int): Number of background samples to use
            
        Returns:
            List: Background data samples
        """
        # Randomly sample from data_list
        if len(data_list) > num_samples:
            indices = np.random.choice(len(data_list), num_samples, replace=False)
            background_data = [data_list[i] for i in indices]
        else:
            background_data = data_list
        
        return background_data
    
    def _model_predict(self, data_batch):
        """
        Wrapper function for model prediction.
        
        Args:
            data_batch: Batch of data
            
        Returns:
            numpy.ndarray: Model predictions
        """
        # Move data to device
        x = data_batch.x.to(self.device)
        edge_index = data_batch.edge_index.to(self.device)
        edge_attr = data_batch.edge_attr.to(self.device) if hasattr(data_batch, 'edge_attr') else None
        batch = data_batch.batch.to(self.device) if hasattr(data_batch, 'batch') else None
        
        # Get predictions
        with torch.no_grad():
            preds = self.model(x, edge_index, edge_attr, batch)
        
        return preds.cpu().numpy()
    
    def compute_node_shap_values(
        self,
        data_list: List,
        counter_names: List[str],
        background_samples: int = 100,
        test_samples: int = 10,
        save_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute SHAP values for node features.
        
        Args:
            data_list (List): List of PyTorch Geometric Data objects
            counter_names (List[str]): List of counter names
            background_samples (int): Number of background samples to use
            test_samples (int): Number of test samples to explain
            save_dir (Optional[str]): Directory to save SHAP plots
            
        Returns:
            Tuple[np.ndarray, List[str]]: SHAP values and feature names
        """
        logger.info(f"Computing SHAP values for {test_samples} samples using {background_samples} background samples")
        
        # Prepare background data
        background_data = self._prepare_background_data(data_list, background_samples)
        
        # Select test samples
        if len(data_list) > test_samples:
            test_indices = np.random.choice(len(data_list), test_samples, replace=False)
            test_data = [data_list[i] for i in test_indices]
        else:
            test_data = data_list
        
        # Create explainer
        explainer = shap.KernelExplainer(self._model_predict, background_data)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(test_data)
        
        # Save SHAP summary plot if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                feature_names=counter_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP summary plot saved to {os.path.join(save_dir, 'shap_summary.png')}")
        
        return shap_values, counter_names
    
    def identify_bottlenecks(
        self,
        shap_values: np.ndarray,
        counter_names: List[str],
        top_n: int = 10,
        save_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Identify I/O bottlenecks based on SHAP values.
        
        Args:
            shap_values (np.ndarray): SHAP values
            counter_names (List[str]): List of counter names
            top_n (int): Number of top bottlenecks to identify
            save_dir (Optional[str]): Directory to save bottleneck plots
            
        Returns:
            pd.DataFrame: DataFrame with bottleneck counters and their importance
        """
        logger.info(f"Identifying top {top_n} I/O bottlenecks")
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame with counter names and importance
        bottlenecks_df = pd.DataFrame({
            'counter': counter_names,
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        bottlenecks_df = bottlenecks_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Get top N bottlenecks
        top_bottlenecks = bottlenecks_df.head(top_n)
        
        # Save bottleneck plot if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            plt.barh(top_bottlenecks['counter'][::-1], top_bottlenecks['importance'][::-1])
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'Top {top_n} I/O Bottlenecks')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'bottlenecks.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save bottlenecks to CSV
            bottlenecks_df.to_csv(os.path.join(save_dir, 'bottlenecks.csv'), index=False)
            
            logger.info(f"Bottleneck analysis saved to {save_dir}")
        
        return bottlenecks_df

def analyze_bottlenecks(
    model: torch.nn.Module,
    dataset,
    counter_names: List[str],
    cfg: DictConfig
) -> pd.DataFrame:
    """
    Analyze I/O bottlenecks using SHAP values.
    
    Args:
        model (torch.nn.Module): Trained GNN model
        dataset: PyTorch Geometric dataset
        counter_names (List[str]): List of counter names
        cfg (DictConfig): Configuration object
        
    Returns:
        pd.DataFrame: DataFrame with bottleneck counters and their importance
    """
    # Initialize explainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explainer = GNNExplainer(model, device)
    
    # Compute SHAP values
    shap_values, _ = explainer.compute_node_shap_values(
        data_list=dataset,
        counter_names=counter_names,
        background_samples=cfg.shap.background_samples,
        test_samples=cfg.shap.test_samples,
        save_dir=cfg.shap.save_dir
    )
    
    # Identify bottlenecks
    bottlenecks_df = explainer.identify_bottlenecks(
        shap_values=shap_values,
        counter_names=counter_names,
        top_n=cfg.shap.top_n,
        save_dir=cfg.shap.save_dir
    )
    
    return bottlenecks_df
