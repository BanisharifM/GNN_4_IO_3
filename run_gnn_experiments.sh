#!/bin/bash
# Master script to run GNN experiments for I/O bottleneck analysis
# This script organizes all commands for different phases of experimentation

# Set default values for common parameters
DATA_FILE="data/split_data/train.csv"
MI_FILE="data/mutual_information2.csv"
BASE_DIR="."
HIDDEN_DIM=64
NUM_LAYERS=2

# Create required directories
mkdir -p ${BASE_DIR}/data/preprocessed
mkdir -p ${BASE_DIR}/logs/training
mkdir -p ${BASE_DIR}/logs/shap
mkdir -p ${BASE_DIR}/logs/hyperopt
mkdir -p ${BASE_DIR}/logs/comparisons

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "A script to run GNN experiments for I/O bottleneck analysis."
    echo ""
    echo "Options:"
    echo "  --data-file FILE       Path to data CSV file (default: $DATA_FILE)"
    echo "  --mi-file FILE         Path to mutual information CSV file (default: $MI_FILE)"
    echo "  --base-dir DIR         Base directory for the project (default: $BASE_DIR)"
    echo "  --hidden-dim NUM       Hidden dimension size (default: $HIDDEN_DIM)"
    echo "  --num-layers NUM       Number of GNN layers (default: $NUM_LAYERS)"
    echo "  -h, --help             Display this help message"
    echo ""
    echo "Commands:"
    echo "  split-data             Split data into train/val/test sets"
    echo "  baseline-gcn           Run baseline GCN model"
    echo "  baseline-gat           Run baseline GAT model"
    echo "  adv-feat-gcn           Run GCN with advanced feature selection"
    echo "  adv-feat-parallel      Run GCN with advanced feature selection and parallel processing"
    echo "  cluster3-gcn           Run GCN with 3-cluster approach"
    echo "  cluster5-gcn           Run GCN with 5-cluster approach"
    echo "  combined               Run combined approach (advanced feature selection + clustering)"
    echo "  hyperopt               Run hyperparameter optimization"
    echo "  compare                Generate comparison report"
    echo "  all                    Run all experiments sequentially"
    echo ""
    echo "Examples:"
    echo "  $0 baseline-gcn                   # Run baseline GCN model with default parameters"
    echo "  $0 --hidden-dim 128 baseline-gat  # Run baseline GAT model with hidden dimension 128"
    echo "  $0 all                            # Run all experiments with default parameters"
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --mi-file)
            MI_FILE="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Function to run data splitting
run_split_data() {
    echo "=== Splitting Data ==="
    python ${BASE_DIR}/scripts/00_split_data.py \
        --data_file ${BASE_DIR}/data/sample_train_100.csv \
        --output_dir ${BASE_DIR}/data/split_data \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15
    
    echo "Data splitting completed."
}

# Function to run baseline GCN model
run_baseline_gcn() {
    echo "=== Running Baseline GCN Model ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/baseline_gcn \
        --use_advanced_feature_selection False \
        --use_clustering False
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gcn \
        --output_dir ${BASE_DIR}/logs/training/baseline_gcn \
        --model_type gcn \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gcn \
        --model_dir ${BASE_DIR}/logs/training/baseline_gcn \
        --output_dir ${BASE_DIR}/logs/shap/baseline_gcn
    
    echo "Baseline GCN model completed."
}

# Function to run baseline GAT model
run_baseline_gat() {
    echo "=== Running Baseline GAT Model ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/baseline_gat \
        --use_advanced_feature_selection False \
        --use_clustering False
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gat \
        --output_dir ${BASE_DIR}/logs/training/baseline_gat \
        --model_type gat \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gat \
        --model_dir ${BASE_DIR}/logs/training/baseline_gat \
        --output_dir ${BASE_DIR}/logs/shap/baseline_gat
    
    echo "Baseline GAT model completed."
}

# Function to run advanced feature selection with GCN
run_adv_feat_gcn() {
    echo "=== Running Advanced Feature Selection with GCN ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/adv_feat_gcn \
        --use_advanced_feature_selection True \
        --use_clustering False
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_gcn \
        --output_dir ${BASE_DIR}/logs/training/adv_feat_gcn \
        --model_type gcn \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_gcn \
        --model_dir ${BASE_DIR}/logs/training/adv_feat_gcn \
        --output_dir ${BASE_DIR}/logs/shap/adv_feat_gcn
    
    echo "Advanced feature selection with GCN completed."
}

# Function to run advanced feature selection with parallel processing
run_adv_feat_parallel() {
    echo "=== Running Advanced Feature Selection with Parallel Processing ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/adv_feat_parallel \
        --use_advanced_feature_selection True \
        --use_parallel True \
        --use_clustering False
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_parallel \
        --output_dir ${BASE_DIR}/logs/training/adv_feat_parallel \
        --model_type gcn \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_parallel \
        --model_dir ${BASE_DIR}/logs/training/adv_feat_parallel \
        --output_dir ${BASE_DIR}/logs/shap/adv_feat_parallel
    
    echo "Advanced feature selection with parallel processing completed."
}

# Function to run clustering approach with 3 clusters
run_cluster3_gcn() {
    echo "=== Running Clustering Approach with 3 Clusters ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/cluster3_gcn \
        --use_advanced_feature_selection False \
        --use_clustering True \
        --n_clusters 3
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster3_gcn \
        --output_dir ${BASE_DIR}/logs/training/cluster3_gcn \
        --model_type gcn \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster3_gcn \
        --model_dir ${BASE_DIR}/logs/training/cluster3_gcn \
        --output_dir ${BASE_DIR}/logs/shap/cluster3_gcn
    
    echo "Clustering approach with 3 clusters completed."
}

# Function to run clustering approach with 5 clusters
run_cluster5_gcn() {
    echo "=== Running Clustering Approach with 5 Clusters ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/cluster5_gcn \
        --use_advanced_feature_selection False \
        --use_clustering True \
        --n_clusters 5
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster5_gcn \
        --output_dir ${BASE_DIR}/logs/training/cluster5_gcn \
        --model_type gcn \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster5_gcn \
        --model_dir ${BASE_DIR}/logs/training/cluster5_gcn \
        --output_dir ${BASE_DIR}/logs/shap/cluster5_gcn
    
    echo "Clustering approach with 5 clusters completed."
}

# Function to run combined approach
run_combined() {
    echo "=== Running Combined Approach (Advanced Feature Selection + Clustering) ==="
    
    echo "Step 1: Preprocessing data"
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${DATA_FILE} \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/combined \
        --use_advanced_feature_selection True \
        --use_clustering True \
        --n_clusters 3 \
        --use_parallel True
    
    echo "Step 2: Training model"
    python ${BASE_DIR}/scripts/02_train_model.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/combined \
        --output_dir ${BASE_DIR}/logs/training/combined \
        --model_type gcn \
        --hidden_dim ${HIDDEN_DIM} \
        --num_layers ${NUM_LAYERS}
    
    echo "Step 3: Analyzing bottlenecks"
    python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/combined \
        --model_dir ${BASE_DIR}/logs/training/combined \
        --output_dir ${BASE_DIR}/logs/shap/combined
    
    echo "Combined approach completed."
}

# Function to run hyperparameter optimization
run_hyperopt() {
    echo "=== Running Hyperparameter Optimization ==="
    
    echo "Running hyperparameter optimization"
    python ${BASE_DIR}/scripts/04_hyperparameter_optimization.py \
        --preprocessed_dir ${BASE_DIR}/data/preprocessed/combined \
        --output_dir ${BASE_DIR}/logs/hyperopt/combined \
        --num_samples 20 \
        --max_epochs 50
    
    echo "Hyperparameter optimization completed."
}

# Function to generate comparison report
run_compare() {
    echo "=== Generating Comparison Report ==="
    
    echo "Generating comparison report"
    python ${BASE_DIR}/scripts/generate_comparison_report.py \
        --experiment_dirs \
        ${BASE_DIR}/logs/training/baseline_gcn \
        ${BASE_DIR}/logs/training/baseline_gat \
        ${BASE_DIR}/logs/training/adv_feat_gcn \
        ${BASE_DIR}/logs/training/cluster3_gcn \
        ${BASE_DIR}/logs/training/combined \
        --output_dir ${BASE_DIR}/logs/comparisons
    
    echo "Comparison report generation completed."
}

# Function to run all experiments
run_all() {
    run_split_data
    run_baseline_gcn
    run_baseline_gat
    run_adv_feat_gcn
    run_adv_feat_parallel
    run_cluster3_gcn
    run_cluster5_gcn
    run_combined
    run_hyperopt
    run_compare
}

# Execute the specified command
case ${COMMAND} in
    split-data)
        run_split_data
        ;;
    baseline-gcn)
        run_baseline_gcn
        ;;
    baseline-gat)
        run_baseline_gat
        ;;
    adv-feat-gcn)
        run_adv_feat_gcn
        ;;
    adv-feat-parallel)
        run_adv_feat_parallel
        ;;
    cluster3-gcn)
        run_cluster3_gcn
        ;;
    cluster5-gcn)
        run_cluster5_gcn
        ;;
    combined)
        run_combined
        ;;
    hyperopt)
        run_hyperopt
        ;;
    compare)
        run_compare
        ;;
    all)
        run_all
        ;;
    *)
        echo "Error: Unknown command '${COMMAND}'"
        show_help
        exit 1
        ;;
esac

exit 0
