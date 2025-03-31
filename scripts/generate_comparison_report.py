import os
import argparse
import logging
import glob
from datetime import datetime
from src.utils.experiment_comparison import ExperimentComparisonManager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """
    Generate a comparison report for multiple experiment approaches.
    
    This script compares the results of different experimental approaches
    (standard GNN, advanced feature selection, clustering, etc.) and
    generates a comprehensive HTML report with visualizations.
    """
    parser = argparse.ArgumentParser(description="Generate comparison report for multiple experiment approaches")
    
    parser.add_argument(
        "--experiment_dirs",
        nargs="+",
        required=True,
        help="List of experiment directories to compare"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/comparisons",
        help="Directory to save the comparison report"
    )
    
    parser.add_argument(
        "--report_name",
        type=str,
        default="approach_comparison",
        help="Name for the comparison report"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create experiment comparison manager
    comparison_manager = ExperimentComparisonManager(
        base_output_dir=args.output_dir,
        report_name=args.report_name
    )
    
    # Generate comparison report
    report_path = comparison_manager.generate_comparison_report(args.experiment_dirs)
    
    if report_path:
        logger.info(f"Comparison report generated successfully at: {report_path}")
        
        # Create symlink to latest report
        latest_link = os.path.join(args.output_dir, "latest_comparison_report.html")
        
        if os.path.exists(latest_link):
            os.remove(latest_link)
        
        try:
            os.symlink(report_path, latest_link)
            logger.info(f"Created symlink to latest report: {latest_link}")
        except Exception as e:
            logger.warning(f"Failed to create symlink: {e}")
    else:
        logger.error("Failed to generate comparison report")

if __name__ == "__main__":
    main()
