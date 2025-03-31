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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VisualizationManager:
    """
    Manager for creating and saving visualizations.
    """
    
    def __init__(self, output_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir (str): Directory to save visualizations
            experiment_name (str, optional): Name of the experiment
        """
        self.output_dir = output_dir
        
        # Create timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name is not None:
            self.experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
        else:
            self.experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories for different types of visualizations
        self.training_dir = os.path.join(self.experiment_dir, "training")
        self.prediction_dir = os.path.join(self.experiment_dir, "predictions")
        self.shap_dir = os.path.join(self.experiment_dir, "shap")
        self.comparison_dir = os.path.join(self.experiment_dir, "comparisons")
        
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)
        os.makedirs(self.shap_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        logger.info(f"Visualization manager initialized with output directory: {self.experiment_dir}")
    
    def plot_training_curves(
        self, 
        train_losses: List[float], 
        val_losses: List[float], 
        best_epoch: int,
        title: str = "Training and Validation Loss",
        filename: Optional[str] = None
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses (List[float]): Training losses
            val_losses (List[float]): Validation losses
            best_epoch (int): Best epoch based on validation loss
            title (str): Plot title
            filename (str, optional): Filename to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        # Mark best epoch
        if best_epoch >= 0:
            plt.axvline(x=best_epoch + 1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch + 1})')
            plt.plot(best_epoch + 1, val_losses[best_epoch], 'go', markersize=8)
        
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.training_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_path}")
    
    def plot_predictions_vs_targets(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        title: str = "Predictions vs Targets",
        filename: Optional[str] = None,
        include_metrics: bool = True
    ):
        """
        Plot predictions vs targets.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            title (str): Plot title
            filename (str, optional): Filename to save the plot
            include_metrics (bool): Whether to include metrics in the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # Create scatter plot
        plt.scatter(targets, predictions, alpha=0.5, edgecolors='w', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(predictions), np.min(targets))
        max_val = max(np.max(predictions), np.max(targets))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.title(title)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        
        # Add metrics to the plot
        if include_metrics:
            metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nCorrelation: {correlation:.4f}'
            plt.annotate(
                metrics_text, 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.prediction_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Predictions vs targets plot saved to {save_path}")
    
    def plot_error_distribution(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        title: str = "Prediction Error Distribution",
        filename: Optional[str] = None
    ):
        """
        Plot distribution of prediction errors.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            title (str): Plot title
            filename (str, optional): Filename to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate errors
        errors = predictions - targets
        
        # Create histogram
        sns.histplot(errors, kde=True)
        
        plt.title(title)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        
        # Add metrics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        metrics_text = f'Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}'
        plt.annotate(
            metrics_text, 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
        
        plt.legend()
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.prediction_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution plot saved to {save_path}")
    
    def plot_shap_summary(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str],
        title: str = "SHAP Summary",
        filename: Optional[str] = None,
        max_display: int = 20
    ):
        """
        Plot SHAP summary.
        
        Args:
            shap_values (np.ndarray): SHAP values
            feature_names (List[str]): Feature names
            title (str): Plot title
            filename (str, optional): Filename to save the plot
            max_display (int): Maximum number of features to display
        """
        plt.figure(figsize=(12, 10))
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        indices = np.argsort(mean_abs_shap)[::-1]
        
        # Limit to max_display
        if len(indices) > max_display:
            indices = indices[:max_display]
        
        # Create bar plot
        plt.barh(
            range(len(indices)), 
            mean_abs_shap[indices], 
            color='skyblue', 
            edgecolor='navy'
        )
        
        # Add feature names
        plt.yticks(
            range(len(indices)), 
            [feature_names[i] for i in indices]
        )
        
        plt.title(title)
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.shap_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {save_path}")
    
    def plot_shap_heatmap(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str],
        title: str = "SHAP Heatmap",
        filename: Optional[str] = None,
        max_display: int = 20
    ):
        """
        Plot SHAP heatmap.
        
        Args:
            shap_values (np.ndarray): SHAP values
            feature_names (List[str]): Feature names
            title (str): Plot title
            filename (str, optional): Filename to save the plot
            max_display (int): Maximum number of features to display
        """
        plt.figure(figsize=(14, 10))
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        indices = np.argsort(mean_abs_shap)[::-1]
        
        # Limit to max_display
        if len(indices) > max_display:
            indices = indices[:max_display]
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom_diverging',
            ['#1E88E5', '#FFFFFF', '#FF0D57']
        )
        
        # Create heatmap
        sns.heatmap(
            shap_values[:, indices].T, 
            cmap=cmap, 
            center=0, 
            yticklabels=[feature_names[i] for i in indices],
            cbar_kws={'label': 'SHAP Value'}
        )
        
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Feature')
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.shap_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP heatmap saved to {save_path}")
    
    def plot_bottleneck_pie(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str],
        title: str = "I/O Bottleneck Distribution",
        filename: Optional[str] = None,
        top_n: int = 10
    ):
        """
        Plot pie chart of top bottlenecks.
        
        Args:
            shap_values (np.ndarray): SHAP values
            feature_names (List[str]): Feature names
            title (str): Plot title
            filename (str, optional): Filename to save the plot
            top_n (int): Number of top bottlenecks to display
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        indices = np.argsort(mean_abs_shap)[::-1]
        
        # Get top N features
        top_indices = indices[:top_n]
        top_values = mean_abs_shap[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Calculate total importance
        total_importance = np.sum(mean_abs_shap)
        
        # Calculate percentages
        percentages = top_values / total_importance * 100
        
        # Create pie chart
        plt.pie(
            percentages, 
            labels=top_names, 
            autopct='%1.1f%%', 
            startangle=90, 
            shadow=True
        )
        
        plt.title(title)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.shap_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bottleneck pie chart saved to {save_path}")
    
    def plot_feature_importance_comparison(
        self,
        importance_dict: Dict[str, Dict[str, float]],
        title: str = "Feature Importance Comparison",
        filename: Optional[str] = None,
        max_display: int = 15
    ):
        """
        Plot comparison of feature importances across different methods.
        
        Args:
            importance_dict (Dict[str, Dict[str, float]]): Dictionary of feature importances
                {method_name: {feature_name: importance}}
            title (str): Plot title
            filename (str, optional): Filename to save the plot
            max_display (int): Maximum number of features to display
        """
        plt.figure(figsize=(14, 10))
        
        # Get all feature names
        all_features = set()
        for method_dict in importance_dict.values():
            all_features.update(method_dict.keys())
        
        # Create DataFrame for plotting
        df = pd.DataFrame(index=all_features)
        
        for method, importances in importance_dict.items():
            df[method] = pd.Series(importances)
        
        # Fill NaN values with 0
        df.fillna(0, inplace=True)
        
        # Calculate mean importance across methods
        df['mean'] = df.mean(axis=1)
        
        # Sort by mean importance
        df.sort_values('mean', ascending=False, inplace=True)
        
        # Limit to max_display
        if len(df) > max_display:
            df = df.iloc[:max_display]
        
        # Drop mean column
        df.drop('mean', axis=1, inplace=True)
        
        # Create bar plot
        df.plot(kind='barh', figsize=(14, 10))
        
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Method')
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.comparison_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance comparison saved to {save_path}")
    
    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        title: str = "Model Performance Comparison",
        filename: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Plot comparison of model performances.
        
        Args:
            results_dict (Dict[str, Dict[str, float]]): Dictionary of model results
                {model_name: {metric_name: value}}
            title (str): Plot title
            filename (str, optional): Filename to save the plot
            metrics (List[str], optional): List of metrics to include
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        # Create DataFrame for plotting
        df = pd.DataFrame(results_dict).T
        
        # Filter metrics
        df = df[metrics]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i] if len(metrics) > 1 else axes
            df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the plot
        if filename is None:
            # Create filename from title
            filename = title.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.png'
        
        save_path = os.path.join(self.comparison_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison saved to {save_path}")
    
    def create_experiment_summary(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        title: str = "Experiment Summary",
        filename: Optional[str] = None
    ):
        """
        Create summary of experiment configuration and results.
        
        Args:
            config (Dict[str, Any]): Experiment configuration
            results (Dict[str, Any]): Experiment results
            title (str): Summary title
            filename (str, optional): Filename to save the summary
        """
        # Create summary dictionary
        summary = {
            'title': title,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': config,
            'results': results
        }
        
        # Save as JSON
        if filename is None:
            filename = 'experiment_summary.json'
        
        save_path = os.path.join(self.experiment_dir, filename)
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Experiment summary saved to {save_path}")
        
        # Also create a human-readable text version
        text_path = os.path.join(self.experiment_dir, filename.replace('.json', '.txt'))
        
        with open(text_path, 'w') as f:
            f.write(f"{title}\n")
            f.write(f"{'=' * len(title)}\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            f.write("Configuration:\n")
            f.write("--------------\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nResults:\n")
            f.write("--------\n")
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Human-readable summary saved to {text_path}")
    
    def create_dashboard(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        plots: List[str],
        title: str = "Experiment Dashboard",
        filename: Optional[str] = None
    ):
        """
        Create dashboard with experiment summary and plots.
        
        Args:
            config (Dict[str, Any]): Experiment configuration
            results (Dict[str, Any]): Experiment results
            plots (List[str]): List of plot filenames to include
            title (str): Dashboard title
            filename (str, optional): Filename to save the dashboard
        """
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Add title
        fig.suptitle(title, fontsize=16)
        
        # Add configuration and results text
        ax_text = fig.add_subplot(gs[0, 0])
        ax_text.axis('off')
        
        text = "Configuration:\n"
        for key, value in config.items():
            text += f"- {key}: {value}\n"
        
        text += "\nResults:\n"
        for key, value in results.items():
            if isinstance(value, dict):
                text += f"- {key}:\n"
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        text += f"  - {subkey}: {subvalue:.4f}\n"
                    else:
                        text += f"  - {subkey}: {subvalue}\n"
            elif isinstance(value, float):
                text += f"- {key}: {value:.4f}\n"
            else:
                text += f"- {key}: {value}\n"
        
        ax_text.text(0, 1, text, verticalalignment='top', fontsize=10)
        
        # Add plots
        plot_positions = [
            (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]
        
        for i, plot_file in enumerate(plots):
            if i >= len(plot_positions):
                break
            
            row, col = plot_positions[i]
            ax = fig.add_subplot(gs[row, col])
            
            try:
                img = plt.imread(plot_file)
                ax.imshow(img)
                ax.set_title(os.path.basename(plot_file).replace('.png', '').replace('_', ' ').title())
                ax.axis('off')
            except Exception as e:
                logger.warning(f"Error loading plot {plot_file}: {e}")
                ax.text(0.5, 0.5, f"Error loading plot:\n{plot_file}", ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the dashboard
        if filename is None:
            filename = 'experiment_dashboard.png'
        
        save_path = os.path.join(self.experiment_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Experiment dashboard saved to {save_path}")
    
    def create_html_report(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        plots: List[str],
        title: str = "Experiment Report",
        filename: Optional[str] = None
    ):
        """
        Create HTML report with experiment summary and plots.
        
        Args:
            config (Dict[str, Any]): Experiment configuration
            results (Dict[str, Any]): Experiment results
            plots (List[str]): List of plot filenames to include
            title (str): Report title
            filename (str, optional): Filename to save the report
        """
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1, h2 {{
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .plot-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
                }}
                .plot {{
                    margin: 10px;
                    text-align: center;
                }}
                .plot img {{
                    max-width: 100%;
                    height: auto;
                    max-height: 400px;
                    border: 1px solid #ddd;
                }}
                .timestamp {{
                    color: #666;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="section">
                    <h2>Configuration</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add configuration
        for key, value in config.items():
            html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Results</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add results
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        html += f"""
                        <tr>
                            <td>{key} - {subkey}</td>
                            <td>{subvalue:.4f}</td>
                        </tr>
                        """
                    else:
                        html += f"""
                        <tr>
                            <td>{key} - {subkey}</td>
                            <td>{subvalue}</td>
                        </tr>
                        """
            elif isinstance(value, float):
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
            else:
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """
        
        html += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="plot-container">
        """
        
        # Add plots
        for plot_file in plots:
            try:
                # Get relative path
                rel_path = os.path.relpath(plot_file, self.experiment_dir)
                plot_name = os.path.basename(plot_file).replace('.png', '').replace('_', ' ').title()
                
                html += f"""
                        <div class="plot">
                            <img src="{rel_path}" alt="{plot_name}">
                            <p>{plot_name}</p>
                        </div>
                """
            except Exception as e:
                logger.warning(f"Error including plot {plot_file}: {e}")
        
        html += """
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        if filename is None:
            filename = 'experiment_report.html'
        
        save_path = os.path.join(self.experiment_dir, filename)
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {save_path}")
        
        return save_path
