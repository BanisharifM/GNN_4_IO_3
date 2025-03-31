# GNN for I/O Bottleneck Analysis

This project implements a Graph Neural Network (GNN) approach for predicting job performance tags and identifying I/O bottlenecks in HPC systems. The implementation treats each job as a graph with I/O counters as nodes and edges based on mutual information.

## Features

- **Graph-based Representation**: Models I/O counters as nodes in a graph with edges based on mutual information
- **GNN Model**: Implements flexible GNN architectures (GCN, GAT) for performance prediction
- **SHAP Analysis**: Identifies I/O bottlenecks using SHAP values
- **Advanced Feature Selection**: Optional Min-max mutual information and Sequential Backward Selection (SBS)
- **Clustering-based Approach**: Optional clustering of jobs with similar I/O characteristics
- **Comprehensive Visualization**: Detailed visualizations and HTML reports
- **Experiment Comparison**: Tools to compare different approaches and configurations
- **HPC Integration**: Slurm scripts for running on HPC clusters

## Project Structure

```
GNN_4_IO_2-main/
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration file
│   ├── dev.yaml                 # Development environment config
│   └── hyperparameters/         # Hyperparameter configurations
│
├── data/                        # Data directory
│   ├── mutual_information2.csv  # Mutual information between counters
│   └── sample_train_100.csv     # Sample training data
│
├── logs/                        # Log directories
│   ├── evaluation/              # Evaluation results
│   ├── shap/                    # SHAP analysis results
│   └── training/                # Training logs and checkpoints
│
├── scripts/                     # Pipeline and utility scripts
│   ├── 00_split_data.py         # Permanent data splitting
│   ├── 01_preprocess_data.py    # Data preprocessing
│   ├── 02_train_model.py        # Model training
│   ├── 03_analyze_bottlenecks.py # Bottleneck analysis
│   ├── 04_hyperparameter_optimization.py # Hyperparameter tuning
│   ├── generate_comparison_report.py # Compare different approaches
│   ├── generate_report.py       # Generate experiment report
│   ├── test_graph_construction.py # Test graph construction
│   ├── test_gnn_training.py     # Test GNN training
│   └── test_shap_analysis.py    # Test SHAP analysis
│
├── slurm/                       # Slurm job scripts
│   ├── 01_split.slurm           # Data splitting job
│   ├── 02_preprocess.slurm      # Preprocessing job
│   ├── 03_train.slurm           # Training job
│   ├── 04_hyperopt.slurm        # Hyperparameter optimization job
│   └── 05_shap.slurm            # SHAP analysis job
│
├── src/                         # Source code
│   ├── data.py                  # Graph construction and dataset
│   ├── main.py                  # Main application
│   ├── models/                  # Model implementations
│   │   └── gnn.py               # GNN model architecture
│   └── utils/                   # Utility modules
│       ├── clustering.py        # Clustering manager
│       ├── data_utils.py        # Data utilities
│       ├── experiment_comparison.py # Experiment comparison
│       ├── feature_selection.py # Advanced feature selection
│       ├── shap_utils.py        # SHAP analysis utilities
│       ├── cluster_training.py  # Cluster-based training
│       └── visualization.py     # Visualization utilities
│
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── run_pipeline.sh              # Master script for running pipeline
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GNN_4_IO_2.git
cd GNN_4_IO_2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step-by-Step Pipeline

The project is designed to be run in steps, which is especially important for handling large datasets:

1. **Split Data** (Optional for large datasets):
```bash
python scripts/00_split_data.py \
  --data_file data/sample_train_100.csv \
  --output_dir data/split_data \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

2. **Preprocess Data**:
```bash
python scripts/01_preprocess_data.py \
  --data_file data/split_data/train.csv \
  --mi_file data/mutual_information2.csv \
  --output_dir data/preprocessed \
  --use_advanced_feature_selection False \
  --use_clustering False
```

3. **Train Model**:
```bash
python scripts/02_train_model.py \
  --preprocessed_dir data/preprocessed \
  --output_dir logs/training \
  --model_type gcn \
  --hidden_dim 64 \
  --num_layers 2
```

4. **Analyze Bottlenecks**:
```bash
python scripts/03_analyze_bottlenecks.py \
  --preprocessed_dir data/preprocessed \
  --model_dir logs/training \
  --output_dir logs/shap
```

5. **Hyperparameter Optimization** (Optional):
```bash
python scripts/04_hyperparameter_optimization.py \
  --preprocessed_dir data/preprocessed \
  --output_dir logs/hyperopt \
  --num_samples 10 \
  --max_epochs 50
```

### Optional Improvements

The implementation includes several optional improvements that can be enabled through configuration:

1. **Advanced Feature Selection**:
   - Enable with `--use_advanced_feature_selection True`
   - Uses Min-max mutual information and Sequential Backward Selection (SBS)
   - Can be run in parallel with `--use_parallel True`

2. **Clustering-based Approach**:
   - Enable with `--use_clustering True`
   - Groups jobs with similar I/O characteristics
   - Trains separate GNN models for each cluster

3. **Experiment Comparison**:
   - Compare different approaches using:
   ```bash
   python scripts/generate_comparison_report.py \
     --experiment_dirs logs/training/exp1 logs/training/exp2 \
     --output_dir logs/comparisons
   ```

### Running on HPC Clusters

For running on HPC clusters like Delta GPU, use the provided Slurm scripts:

```bash
# Submit data splitting job
sbatch slurm/01_split.slurm

# Submit preprocessing job
sbatch slurm/02_preprocess.slurm

# Submit training job
sbatch slurm/03_train.slurm

# Submit hyperparameter optimization job
sbatch slurm/04_hyperopt.slurm

# Submit SHAP analysis job
sbatch slurm/05_shap.slurm
```

## Configuration

The project uses a flexible configuration system with several options:

### Graph Construction
- `mi_threshold`: Threshold for mutual information (default: 0.3259)
- `use_advanced_feature_selection`: Whether to use advanced feature selection
- `n_features_to_select`: Number of features to select if using advanced selection
- `use_parallel`: Whether to use parallel processing for feature selection

### Clustering
- `use_clustering`: Whether to use clustering
- `n_clusters`: Number of clusters if using clustering

### GNN Model
- `model_type`: Type of GNN model ('gcn' or 'gat')
- `hidden_dim`: Hidden dimension size
- `num_layers`: Number of GNN layers
- `dropout`: Dropout rate

### Training
- `learning_rate`: Learning rate for optimizer
- `weight_decay`: Weight decay for optimizer
- `batch_size`: Batch size
- `epochs`: Number of epochs
- `early_stopping_patience`: Patience for early stopping

## Visualization and Reporting

The project includes comprehensive visualization and reporting capabilities:

1. **Training Curves**: Visualize training and validation loss
2. **Predictions vs Targets**: Compare model predictions with ground truth
3. **Error Distribution**: Analyze prediction error distribution
4. **SHAP Summary**: Visualize feature importance based on SHAP values
5. **Bottleneck Pie Chart**: Show distribution of top bottlenecks
6. **Experiment Comparison**: Compare different approaches and configurations
7. **HTML Reports**: Generate detailed HTML reports with all visualizations

## Extending the Project

To extend the project with new features:

1. **New GNN Architectures**: Add new model types in `src/models/gnn.py`
2. **Additional Feature Selection Methods**: Extend `src/utils/feature_selection.py`
3. **New Clustering Algorithms**: Modify `src/utils/clustering.py`
4. **Custom Visualizations**: Add to `src/utils/visualization.py`

## References

This implementation is based on research in GNN-based I/O performance analysis and bottleneck identification, including insights from:

1. "Towards HPC I/O Performance Prediction Through Large-scale Log Analysis" (HPDC '20)
2. "In-depth Pattern Analysis of I/O Behaviors for Adaptive Optimization" (HiPC '21)
