import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import json
import glob
from src.utils.visualization import VisualizationManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ExperimentComparisonManager:
    """
    Manager for comparing different experiment configurations.
    
    This class helps analyze and visualize the differences between various
    experimental approaches (standard GNN, advanced feature selection, clustering, etc.)
    """
    
    def __init__(self, base_output_dir: str, report_name: str = "approach_comparison"):
        """
        Initialize the experiment comparison manager.
        
        Args:
            base_output_dir (str): Base directory for experiment outputs
            report_name (str): Name for the comparison report
        """
        self.base_output_dir = base_output_dir
        self.report_name = report_name
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(base_output_dir, f"{report_name}_{timestamp}")
        
        # Create directory
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize visualization manager
        self.viz_manager = VisualizationManager(self.report_dir, "approach_comparison")
        
        logger.info(f"Experiment comparison manager initialized with output directory: {self.report_dir}")
    
    def collect_experiment_results(self, experiment_dirs: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Collect results from multiple experiments.
        
        Args:
            experiment_dirs (List[str]): List of experiment directories
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of experiment results
        """
        logger.info(f"Collecting results from {len(experiment_dirs)} experiments")
        
        results = {}
        
        for exp_dir in experiment_dirs:
            # Get experiment name from directory
            exp_name = os.path.basename(exp_dir)
            
            # Look for summary files
            summary_files = glob.glob(os.path.join(exp_dir, "*summary*.json"))
            
            if summary_files:
                # Use the first summary file
                with open(summary_files[0], 'r') as f:
                    summary = json.load(f)
                
                # Extract configuration and results
                config = summary.get('config', {})
                exp_results = summary.get('results', {})
                
                # Store in results dictionary
                results[exp_name] = {
                    'config': config,
                    'results': exp_results
                }
            else:
                # Look for individual result files
                result_files = glob.glob(os.path.join(exp_dir, "*test_results*.pkl"))
                
                if result_files:
                    import pickle
                    
                    # Use the first result file
                    with open(result_files[0], 'rb') as f:
                        exp_results = pickle.load(f)
                    
                    # Try to find configuration
                    config_files = glob.glob(os.path.join(exp_dir, "*config*.yaml"))
                    config = {}
                    
                    if config_files:
                        import yaml
                        
                        with open(config_files[0], 'r') as f:
                            config = yaml.safe_load(f)
                    
                    # Store in results dictionary
                    results[exp_name] = {
                        'config': config,
                        'results': exp_results
                    }
                else:
                    logger.warning(f"No results found in {exp_dir}")
        
        logger.info(f"Collected results from {len(results)} experiments")
        
        return results
    
    def compare_performance_metrics(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare performance metrics across experiments.
        
        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of experiment results
            
        Returns:
            pd.DataFrame: DataFrame with performance metrics
        """
        logger.info("Comparing performance metrics across experiments")
        
        # Create DataFrame for metrics
        metrics_df = pd.DataFrame()
        
        for exp_name, exp_data in results.items():
            exp_results = exp_data['results']
            
            # Extract metrics
            metrics = {}
            
            # Handle different result formats
            if isinstance(exp_results, dict):
                # Check if results contain metrics directly
                if 'rmse' in exp_results:
                    metrics['rmse'] = exp_results['rmse']
                    metrics['mae'] = exp_results.get('mae', np.nan)
                    metrics['r2'] = exp_results.get('r2', np.nan)
                    metrics['mse'] = exp_results.get('mse', np.nan)
                # Check if results contain nested metrics
                elif any(isinstance(v, dict) and 'rmse' in v for v in exp_results.values()):
                    # Find the first dictionary with metrics
                    for key, value in exp_results.items():
                        if isinstance(value, dict) and 'rmse' in value:
                            metrics['rmse'] = value['rmse']
                            metrics['mae'] = value.get('mae', np.nan)
                            metrics['r2'] = value.get('r2', np.nan)
                            metrics['mse'] = value.get('mse', np.nan)
                            break
            
            # Add experiment configuration flags
            config = exp_data['config']
            
            # Extract configuration flags
            if isinstance(config, dict):
                # Look for configuration flags in different possible locations
                if 'use_advanced_feature_selection' in config:
                    metrics['use_advanced_feature_selection'] = config['use_advanced_feature_selection']
                elif 'feature_selection' in config and isinstance(config['feature_selection'], dict):
                    metrics['use_advanced_feature_selection'] = config['feature_selection'].get('use_advanced', False)
                
                if 'use_clustering' in config:
                    metrics['use_clustering'] = config['use_clustering']
                elif 'clustering' in config and isinstance(config['clustering'], dict):
                    metrics['use_clustering'] = config['clustering'].get('enabled', False)
                
                if 'use_parallel' in config:
                    metrics['use_parallel'] = config['use_parallel']
                elif 'feature_selection' in config and isinstance(config['feature_selection'], dict):
                    metrics['use_parallel'] = config['feature_selection'].get('use_parallel', False)
            
            # Add to DataFrame
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics], index=[exp_name])], axis=0)
        
        return metrics_df
    
    def compare_feature_importances(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compare feature importances across experiments.
        
        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of experiment results
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of feature importances
        """
        logger.info("Comparing feature importances across experiments")
        
        importances = {}
        
        for exp_name, exp_data in results.items():
            exp_results = exp_data['results']
            
            # Extract feature importances
            if isinstance(exp_results, dict):
                # Check if results contain feature importances
                if 'feature_importances' in exp_results:
                    importances[exp_name] = exp_results['feature_importances']
                # Check if results contain nested feature importances
                elif any(isinstance(v, dict) and 'feature_importances' in v for v in exp_results.values()):
                    # Find the first dictionary with feature importances
                    for key, value in exp_results.items():
                        if isinstance(value, dict) and 'feature_importances' in value:
                            importances[exp_name] = value['feature_importances']
                            break
        
        return importances
    
    def compare_training_times(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare training times across experiments.
        
        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of experiment results
            
        Returns:
            pd.DataFrame: DataFrame with training times
        """
        logger.info("Comparing training times across experiments")
        
        # Create DataFrame for training times
        times_df = pd.DataFrame(columns=['training_time', 'use_advanced_feature_selection', 'use_clustering', 'use_parallel'])
        
        for exp_name, exp_data in results.items():
            exp_results = exp_data['results']
            
            # Extract training time
            training_time = None
            
            if isinstance(exp_results, dict):
                # Check if results contain training time directly
                if 'training_time' in exp_results:
                    training_time = exp_results['training_time']
                # Check if results contain nested training time
                elif any(isinstance(v, dict) and 'training_time' in v for v in exp_results.values()):
                    # Find the first dictionary with training time
                    for key, value in exp_results.items():
                        if isinstance(value, dict) and 'training_time' in value:
                            training_time = value['training_time']
                            break
            
            if training_time is not None:
                # Add experiment configuration flags
                config = exp_data['config']
                
                # Extract configuration flags
                use_advanced = False
                use_clustering = False
                use_parallel = False
                
                if isinstance(config, dict):
                    # Look for configuration flags in different possible locations
                    if 'use_advanced_feature_selection' in config:
                        use_advanced = config['use_advanced_feature_selection']
                    elif 'feature_selection' in config and isinstance(config['feature_selection'], dict):
                        use_advanced = config['feature_selection'].get('use_advanced', False)
                    
                    if 'use_clustering' in config:
                        use_clustering = config['use_clustering']
                    elif 'clustering' in config and isinstance(config['clustering'], dict):
                        use_clustering = config['clustering'].get('enabled', False)
                    
                    if 'use_parallel' in config:
                        use_parallel = config['use_parallel']
                    elif 'feature_selection' in config and isinstance(config['feature_selection'], dict):
                        use_parallel = config['feature_selection'].get('use_parallel', False)
                
                # Add to DataFrame
                times_df.loc[exp_name] = {
                    'training_time': training_time,
                    'use_advanced_feature_selection': use_advanced,
                    'use_clustering': use_clustering,
                    'use_parallel': use_parallel
                }
        
        return times_df
    
    def generate_comparison_report(self, experiment_dirs: List[str]) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            experiment_dirs (List[str]): List of experiment directories
            
        Returns:
            str: Path to the generated report
        """
        logger.info("Generating comparison report")
        
        # Collect results
        results = self.collect_experiment_results(experiment_dirs)
        
        if not results:
            logger.warning("No results found, cannot generate report")
            return None
        
        # Compare performance metrics
        metrics_df = self.compare_performance_metrics(results)
        
        # Compare feature importances
        importances = self.compare_feature_importances(results)
        
        # Compare training times
        times_df = self.compare_training_times(results)
        
        # Save comparison data
        metrics_df.to_csv(os.path.join(self.report_dir, "performance_metrics.csv"))
        
        with open(os.path.join(self.report_dir, "feature_importances.json"), 'w') as f:
            json.dump(importances, f, indent=4)
        
        times_df.to_csv(os.path.join(self.report_dir, "training_times.csv"))
        
        # Create visualizations
        
        # 1. Performance metrics comparison
        if not metrics_df.empty:
            # Create bar plot for each metric
            for metric in ['rmse', 'mae', 'r2']:
                if metric in metrics_df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Sort by metric value
                    sorted_df = metrics_df.sort_values(metric)
                    
                    # Create bar plot
                    ax = sorted_df[metric].plot(kind='bar')
                    
                    plt.title(f"{metric.upper()} Comparison Across Approaches")
                    plt.xlabel("Approach")
                    plt.ylabel(metric.upper())
                    plt.grid(True, alpha=0.3)
                    
                    # Add configuration flags as annotations
                    for i, exp_name in enumerate(sorted_df.index):
                        flags = []
                        
                        if 'use_advanced_feature_selection' in sorted_df.columns and sorted_df.loc[exp_name, 'use_advanced_feature_selection']:
                            flags.append("Adv. Feat.")
                        
                        if 'use_clustering' in sorted_df.columns and sorted_df.loc[exp_name, 'use_clustering']:
                            flags.append("Clust.")
                        
                        if 'use_parallel' in sorted_df.columns and sorted_df.loc[exp_name, 'use_parallel']:
                            flags.append("Parallel")
                        
                        if flags:
                            ax.annotate(
                                ", ".join(flags),
                                xy=(i, sorted_df.loc[exp_name, metric]),
                                xytext=(0, 10),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                rotation=90,
                                fontsize=8
                            )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.report_dir, f"{metric}_comparison.png"), dpi=300)
                    plt.close()
            
            # Create combined metrics plot
            if all(metric in metrics_df.columns for metric in ['rmse', 'mae']):
                plt.figure(figsize=(14, 8))
                
                # Normalize metrics for comparison
                normalized_df = metrics_df.copy()
                
                for metric in ['rmse', 'mae']:
                    if metric in normalized_df.columns:
                        max_val = normalized_df[metric].max()
                        if max_val > 0:
                            normalized_df[metric] = normalized_df[metric] / max_val
                
                if 'r2' in normalized_df.columns:
                    # R² is better when higher, so normalize differently
                    min_val = normalized_df['r2'].min()
                    max_val = normalized_df['r2'].max()
                    if max_val > min_val:
                        normalized_df['r2'] = (normalized_df['r2'] - min_val) / (max_val - min_val)
                
                # Create grouped bar plot
                normalized_df[['rmse', 'mae', 'r2']].plot(kind='bar', figsize=(14, 8))
                
                plt.title("Normalized Metrics Comparison Across Approaches")
                plt.xlabel("Approach")
                plt.ylabel("Normalized Value")
                plt.grid(True, alpha=0.3)
                plt.legend(title="Metric")
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_dir, "normalized_metrics_comparison.png"), dpi=300)
                plt.close()
        
        # 2. Feature importance comparison
        if importances:
            self.viz_manager.plot_feature_importance_comparison(
                importances,
                title="Feature Importance Comparison Across Approaches"
            )
        
        # 3. Training time comparison
        if not times_df.empty:
            plt.figure(figsize=(12, 6))
            
            # Sort by training time
            sorted_df = times_df.sort_values('training_time')
            
            # Create bar plot
            ax = sorted_df['training_time'].plot(kind='bar')
            
            plt.title("Training Time Comparison Across Approaches")
            plt.xlabel("Approach")
            plt.ylabel("Training Time (seconds)")
            plt.grid(True, alpha=0.3)
            
            # Add configuration flags as annotations
            for i, exp_name in enumerate(sorted_df.index):
                flags = []
                
                if sorted_df.loc[exp_name, 'use_advanced_feature_selection']:
                    flags.append("Adv. Feat.")
                
                if sorted_df.loc[exp_name, 'use_clustering']:
                    flags.append("Clust.")
                
                if sorted_df.loc[exp_name, 'use_parallel']:
                    flags.append("Parallel")
                
                if flags:
                    ax.annotate(
                        ", ".join(flags),
                        xy=(i, sorted_df.loc[exp_name, 'training_time']),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        rotation=90,
                        fontsize=8
                    )
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_dir, "training_time_comparison.png"), dpi=300)
            plt.close()
            
            # Create scatter plot of training time vs. performance
            if 'rmse' in metrics_df.columns:
                plt.figure(figsize=(10, 8))
                
                # Merge DataFrames
                merged_df = pd.merge(times_df, metrics_df, left_index=True, right_index=True)
                
                # Create scatter plot
                scatter = plt.scatter(
                    merged_df['training_time'],
                    merged_df['rmse'],
                    c=merged_df['use_clustering'].astype(int) + 2 * merged_df['use_advanced_feature_selection'].astype(int),
                    cmap='viridis',
                    s=100,
                    alpha=0.7
                )
                
                # Add labels for each point
                for i, exp_name in enumerate(merged_df.index):
                    plt.annotate(
                        exp_name,
                        xy=(merged_df.loc[exp_name, 'training_time'], merged_df.loc[exp_name, 'rmse']),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8
                    )
                
                plt.title("Training Time vs. RMSE")
                plt.xlabel("Training Time (seconds)")
                plt.ylabel("RMSE")
                plt.grid(True, alpha=0.3)
                
                # Add colorbar legend
                cbar = plt.colorbar(scatter)
                cbar.set_ticks([0, 1, 2, 3])
                cbar.set_ticklabels(['Basic', 'Clustering', 'Adv. Features', 'Adv. Features + Clustering'])
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_dir, "training_time_vs_rmse.png"), dpi=300)
                plt.close()
        
        # 4. Configuration comparison
        if not metrics_df.empty:
            # Extract configuration flags
            config_cols = [col for col in metrics_df.columns if col.startswith('use_')]
            
            if config_cols:
                # Create heatmap of configurations
                plt.figure(figsize=(12, len(metrics_df) * 0.5 + 2))
                
                # Create heatmap
                sns.heatmap(
                    metrics_df[config_cols].astype(int),
                    cmap='viridis',
                    cbar=False,
                    annot=True,
                    fmt='d',
                    linewidths=0.5
                )
                
                plt.title("Configuration Comparison Across Approaches")
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_dir, "configuration_comparison.png"), dpi=300)
                plt.close()
        
        # Create HTML report
        plots = glob.glob(os.path.join(self.report_dir, "*.png"))
        
        # Create summary of results
        summary = {
            "experiments": list(results.keys()),
            "metrics": metrics_df.to_dict() if not metrics_df.empty else {},
            "training_times": times_df.to_dict() if not times_df.empty else {}
        }
        
        report_path = self.viz_manager.create_html_report(
            config={"experiments": experiment_dirs},
            results=summary,
            plots=plots,
            title="Approach Comparison Report"
        )
        
        logger.info(f"Comparison report generated at {report_path}")
        
        return report_path
