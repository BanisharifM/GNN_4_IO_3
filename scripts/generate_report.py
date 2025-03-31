#!/usr/bin/env python
# Script to generate a comprehensive report from experiment results

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from src.utils.visualization import VisualizationManager

def generate_report(
    experiment_dir,
    preprocessed_dir,
    model_dir,
    shap_dir,
    output_dir=None
):
    """
    Generate a comprehensive HTML report from experiment results.
    
    Args:
        experiment_dir (str): Main experiment directory
        preprocessed_dir (str): Directory with preprocessed data
        model_dir (str): Directory with trained model
        shap_dir (str): Directory with SHAP analysis results
        output_dir (str, optional): Directory to save report
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "report")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization manager
    viz_manager = VisualizationManager(output_dir, "final_report")
    
    # Collect configuration information
    config = {}
    
    # Check for job info files
    job_info_file = os.path.join(experiment_dir, "job_info.txt")
    if os.path.exists(job_info_file):
        with open(job_info_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    config[key.strip()] = value.strip()
    
    # Check for training results
    training_results_file = os.path.join(model_dir, "training_results.pkl")
    training_metrics = {}
    training_plots = []
    
    if os.path.exists(training_results_file):
        import pickle
        with open(training_results_file, 'rb') as f:
            training_results = pickle.load(f)
        
        # Extract metrics
        for key in ['best_epoch', 'best_val_loss', 'test_loss', 'test_rmse', 'training_time']:
            if key in training_results:
                training_metrics[key] = training_results[key]
        
        # Create training plots
        if 'train_losses' in training_results and 'val_losses' in training_results:
            plot_path = viz_manager.plot_training_curves(
                training_results['train_losses'],
                training_results['val_losses'],
                training_results['best_epoch']
            )
            training_plots.append(plot_path)
    
    # Check for prediction results
    predictions_file = os.path.join(model_dir, "predictions.pkl")
    evaluation_plots = []
    
    if os.path.exists(predictions_file):
        import pickle
        with open(predictions_file, 'rb') as f:
            predictions_data = pickle.load(f)
        
        if 'predictions' in predictions_data and 'targets' in predictions_data:
            # Create predictions vs targets plot
            plot_path = viz_manager.plot_predictions_vs_targets(
                predictions_data['predictions'],
                predictions_data['targets']
            )
            evaluation_plots.append(plot_path)
            
            # Create error distribution plot
            plot_path = viz_manager.plot_error_distribution(
                predictions_data['predictions'],
                predictions_data['targets']
            )
            evaluation_plots.append(plot_path)
    
    # Check for SHAP results
    bottlenecks_file = os.path.join(shap_dir, "bottlenecks.csv")
    shap_plots = []
    bottlenecks_data = None
    
    if os.path.exists(bottlenecks_file):
        bottlenecks_df = pd.read_csv(bottlenecks_file)
        bottlenecks_data = bottlenecks_df
        
        # Create bottlenecks pie chart
        plot_path = viz_manager.plot_shap_bottlenecks_pie(
            bottlenecks_df,
            top_n=10
        )
        shap_plots.append(plot_path)
    
    # Check for SHAP values
    shap_values_file = os.path.join(shap_dir, "shap_values.pkl")
    
    if os.path.exists(shap_values_file):
        import pickle
        with open(shap_values_file, 'rb') as f:
            shap_data = pickle.load(f)
        
        if 'shap_values' in shap_data and 'feature_names' in shap_data:
            # Create SHAP summary plot
            plot_path = viz_manager.plot_shap_summary(
                shap_data['shap_values'],
                shap_data['feature_names'],
                top_n=10
            )
            shap_plots.append(plot_path)
            
            # Create SHAP heatmap
            plot_path = viz_manager.plot_shap_heatmap(
                shap_data['shap_values'],
                shap_data['feature_names'],
                top_n=10,
                sample_n=20
            )
            shap_plots.append(plot_path)
    
    # Combine all metrics
    all_metrics = {**training_metrics}
    
    # Save metrics
    metrics_path = viz_manager.save_metrics(all_metrics)
    
    # Create report data
    report_data = {
        'config': config,
        'metrics': all_metrics,
        'training_plots': training_plots,
        'evaluation_plots': evaluation_plots,
        'shap_plots': shap_plots
    }
    
    if bottlenecks_data is not None:
        report_data['bottlenecks'] = {
            'counter': bottlenecks_data['counter'].tolist(),
            'importance': bottlenecks_data['importance'].tolist()
        }
    
    # Generate HTML report
    report_path = viz_manager.create_report(report_data)
    
    logger.info(f"Report generated at {report_path}")
    
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive report from experiment results")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Main experiment directory")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with trained model")
    parser.add_argument("--shap_dir", type=str, required=True, help="Directory with SHAP analysis results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save report")
    
    args = parser.parse_args()
    
    report_path = generate_report(
        args.experiment_dir,
        args.preprocessed_dir,
        args.model_dir,
        args.shap_dir,
        args.output_dir
    )
    
    print(f"Report generated at: {report_path}")
