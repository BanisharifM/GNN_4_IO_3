#!/bin/bash
# Master script to run GNN experiments for I/O bottleneck analysis
# This script allows running:
# 1. Complete experiments
# 2. Individual steps within experiments
# 3. Individual scripts with custom parameters
# 4. Split-based workflow (split data first, then process each split separately)

# Set default values for common parameters
DATA_FILE="data/split_data/train.csv"
MI_FILE="data/mutual_information2.csv"
BASE_DIR="."
HIDDEN_DIM=64
NUM_LAYERS=2
SAMPLE_DIR="sample_100"  # Directory for sample data splits

# Create required directories
mkdir -p ${BASE_DIR}/data/preprocessed
mkdir -p ${BASE_DIR}/logs/training
mkdir -p ${BASE_DIR}/logs/shap
mkdir -p ${BASE_DIR}/logs/hyperopt
mkdir -p ${BASE_DIR}/logs/comparisons
mkdir -p ${BASE_DIR}/data/split_data/${SAMPLE_DIR}

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS] COMMAND [STEP_NUMBER|SCRIPT_ARGS]"
    echo ""
    echo "A script to run GNN experiments for I/O bottleneck analysis."
    echo ""
    echo "Options:"
    echo "  --data-file FILE       Path to data CSV file (default: $DATA_FILE)"
    echo "  --mi-file FILE         Path to mutual information CSV file (default: $MI_FILE)"
    echo "  --base-dir DIR         Base directory for the project (default: $BASE_DIR)"
    echo "  --hidden-dim NUM       Hidden dimension size (default: $HIDDEN_DIM)"
    echo "  --num-layers NUM       Number of GNN layers (default: $NUM_LAYERS)"
    echo "  --sample-dir DIR       Directory for sample data splits (default: $SAMPLE_DIR)"
    echo "  -h, --help             Display this help message"
    echo ""
    echo "Commands for running complete experiments:"
    echo "  baseline-gcn [STEP]    Run baseline GCN model (all steps or specific step 1-3)"
    echo "  baseline-gat [STEP]    Run baseline GAT model (all steps or specific step 1-3)"
    echo "  adv-feat-gcn [STEP]    Run GCN with advanced feature selection (all steps or specific step 1-3)"
    echo "  adv-feat-parallel [STEP] Run GCN with advanced feature selection and parallel processing (all steps or specific step 1-3)"
    echo "  cluster3-gcn [STEP]    Run GCN with 3-cluster approach (all steps or specific step 1-3)"
    echo "  cluster5-gcn [STEP]    Run GCN with 5-cluster approach (all steps or specific step 1-3)"
    echo "  combined [STEP]        Run combined approach (all steps or specific step 1-3)"
    echo "  hyperopt              Run hyperparameter optimization"
    echo "  compare               Generate comparison report"
    echo "  all                   Run all experiments sequentially"
    echo ""
    echo "  For experiment commands, you can specify a step number (1, 2, or 3) to run only that step:"
    echo "    Step 1: Preprocessing data"
    echo "    Step 2: Training model"
    echo "    Step 3: Analyzing bottlenecks"
    echo ""
    echo "Commands for split-based workflow:"
    echo "  split-data            Split data into train/val/test sets"
    echo "  process-splits EXP_NAME Process train/val/test splits separately for a specific experiment"
    echo "                        EXP_NAME can be one of: baseline_gcn, baseline_gat, adv_feat_gcn,"
    echo "                        adv_feat_parallel, cluster3_gcn, cluster5_gcn, combined"
    echo ""
    echo "Commands for running individual scripts:"
    echo "  run-script SCRIPT_NAME [SCRIPT_ARGS]  Run an individual script with custom arguments"
    echo "                                        SCRIPT_NAME can be one of:"
    echo "                                        - 00_split_data"
    echo "                                        - 01_preprocess_data"
    echo "                                        - 02_train_model"
    echo "                                        - 03_analyze_bottlenecks"
    echo "                                        - 04_hyperparameter_optimization"
    echo "                                        - generate_comparison_report"
    echo ""
    echo "Examples:"
    echo "  $0 baseline-gcn                   # Run all steps of baseline GCN model"
    echo "  $0 baseline-gcn 1                 # Run only step 1 (preprocessing) of baseline GCN model"
    echo "  $0 baseline-gcn 2                 # Run only step 2 (training) of baseline GCN model"
    echo "  $0 --hidden-dim 128 baseline-gat  # Run baseline GAT model with hidden dimension 128"
    echo "  $0 all                            # Run all experiments with default parameters"
    echo "  $0 split-data                     # Split data into train/val/test sets"
    echo "  $0 process-splits baseline_gcn    # Process train/val/test splits for baseline GCN"
    echo "  $0 run-script 01_preprocess_data --data_file data/split_data/train.csv --mi_file data/mutual_information2.csv --output_dir data/preprocessed/baseline_gcn"
    echo "                                    # Run 01_preprocess_data.py with custom arguments"
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
        --sample-dir)
            SAMPLE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        run-script)
            COMMAND="run-script"
            SCRIPT_NAME="$2"
            shift 2
            SCRIPT_ARGS="$@"
            break
            ;;
        process-splits)
            COMMAND="process-splits"
            EXP_NAME="$2"
            shift 2
            ;;
        *)
            COMMAND="$1"
            shift
            STEP="$1"
            shift
            ;;
    esac
done

# Function to run data splitting
run_split_data() {
    echo "=== Splitting Data ==="
    python ${BASE_DIR}/scripts/00_split_data.py \
        --data_file ${BASE_DIR}/data/sample_train_100.csv \
        --output_dir ${BASE_DIR}/data/split_data/${SAMPLE_DIR} \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15
    
    echo "Data splitting completed. Files saved to ${BASE_DIR}/data/split_data/${SAMPLE_DIR}/"
}

# Function to process train/val/test splits separately for a specific experiment
process_splits() {
    local exp_name="$1"
    local use_adv_feat="False"
    local use_clustering="False"
    local use_parallel="False"
    local n_clusters=3
    
    # Set experiment-specific parameters
    case ${exp_name} in
        baseline_gcn|baseline_gat)
            use_adv_feat="False"
            use_clustering="False"
            ;;
        adv_feat_gcn)
            use_adv_feat="True"
            use_clustering="False"
            ;;
        adv_feat_parallel)
            use_adv_feat="True"
            use_clustering="False"
            use_parallel="True"
            ;;
        cluster3_gcn)
            use_adv_feat="False"
            use_clustering="True"
            n_clusters=3
            ;;
        cluster5_gcn)
            use_adv_feat="False"
            use_clustering="True"
            n_clusters=5
            ;;
        combined)
            use_adv_feat="True"
            use_clustering="True"
            use_parallel="True"
            n_clusters=3
            ;;
        *)
            echo "Error: Unknown experiment name '${exp_name}'"
            exit 1
            ;;
    esac
    
    echo "=== Processing Splits for ${exp_name} ==="
    
    # Create output directories
    mkdir -p ${BASE_DIR}/data/preprocessed/${exp_name}/train
    mkdir -p ${BASE_DIR}/data/preprocessed/${exp_name}/val
    mkdir -p ${BASE_DIR}/data/preprocessed/${exp_name}/test
    
    # Process training data
    echo "Processing training data..."
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${BASE_DIR}/data/split_data/${SAMPLE_DIR}/train.csv \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/${exp_name}/train \
        --split_type train \
        --use_advanced_feature_selection ${use_adv_feat} \
        --use_clustering ${use_clustering} \
        --use_parallel ${use_parallel} \
        --n_clusters ${n_clusters}
    
    # Process validation data
    echo "Processing validation data..."
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${BASE_DIR}/data/split_data/${SAMPLE_DIR}/val.csv \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/${exp_name}/val \
        --split_type val \
        --use_advanced_feature_selection ${use_adv_feat} \
        --use_clustering ${use_clustering} \
        --use_parallel ${use_parallel} \
        --n_clusters ${n_clusters}
    
    # Process test data
    echo "Processing test data..."
    python ${BASE_DIR}/scripts/01_preprocess_data.py \
        --data_file ${BASE_DIR}/data/split_data/${SAMPLE_DIR}/test.csv \
        --mi_file ${MI_FILE} \
        --output_dir ${BASE_DIR}/data/preprocessed/${exp_name}/test \
        --split_type test \
        --use_advanced_feature_selection ${use_adv_feat} \
        --use_clustering ${use_clustering} \
        --use_parallel ${use_parallel} \
        --n_clusters ${n_clusters}
    
    echo "Split processing completed for ${exp_name}."
    
    # If it's a baseline experiment, also run the training and analysis steps
    if [[ ${exp_name} == baseline_gcn || ${exp_name} == baseline_gat ]]; then
        local model_type=${exp_name#baseline_}  # Remove 'baseline_' prefix to get model type
        
        echo "Running training for ${exp_name}..."
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/${exp_name} \
            --output_dir ${BASE_DIR}/logs/training/${exp_name} \
            --model_type ${model_type} \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        
        echo "Running bottleneck analysis for ${exp_name}..."
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/${exp_name} \
            --model_dir ${BASE_DIR}/logs/training/${exp_name} \
            --output_dir ${BASE_DIR}/logs/shap/${exp_name}
        
        echo "Complete workflow for ${exp_name} finished."
    fi
}

# Function to run baseline GCN model
run_baseline_gcn() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Baseline GCN Model - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/baseline_gcn \
            --use_advanced_feature_selection False \
            --use_clustering False
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Baseline GCN Model - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gcn \
            --output_dir ${BASE_DIR}/logs/training/baseline_gcn \
            --model_type gcn \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Baseline GCN Model - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gcn \
            --model_dir ${BASE_DIR}/logs/training/baseline_gcn \
            --output_dir ${BASE_DIR}/logs/shap/baseline_gcn
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Baseline GCN model completed."
    else
        echo "Baseline GCN model - Step $step completed."
    fi
}

# Function to run baseline GAT model
run_baseline_gat() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Baseline GAT Model - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/baseline_gat \
            --use_advanced_feature_selection False \
            --use_clustering False
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Baseline GAT Model - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gat \
            --output_dir ${BASE_DIR}/logs/training/baseline_gat \
            --model_type gat \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Baseline GAT Model - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/baseline_gat \
            --model_dir ${BASE_DIR}/logs/training/baseline_gat \
            --output_dir ${BASE_DIR}/logs/shap/baseline_gat
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Baseline GAT model completed."
    else
        echo "Baseline GAT model - Step $step completed."
    fi
}

# Function to run advanced feature selection with GCN
run_adv_feat_gcn() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Advanced Feature Selection with GCN - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/adv_feat_gcn \
            --use_advanced_feature_selection True \
            --use_clustering False
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Advanced Feature Selection with GCN - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_gcn \
            --output_dir ${BASE_DIR}/logs/training/adv_feat_gcn \
            --model_type gcn \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Advanced Feature Selection with GCN - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_gcn \
            --model_dir ${BASE_DIR}/logs/training/adv_feat_gcn \
            --output_dir ${BASE_DIR}/logs/shap/adv_feat_gcn
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Advanced feature selection with GCN completed."
    else
        echo "Advanced feature selection with GCN - Step $step completed."
    fi
}

# Function to run advanced feature selection with parallel processing
run_adv_feat_parallel() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Advanced Feature Selection with Parallel Processing - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/adv_feat_parallel \
            --use_advanced_feature_selection True \
            --use_parallel True \
            --use_clustering False
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Advanced Feature Selection with Parallel Processing - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_parallel \
            --output_dir ${BASE_DIR}/logs/training/adv_feat_parallel \
            --model_type gcn \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Advanced Feature Selection with Parallel Processing - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/adv_feat_parallel \
            --model_dir ${BASE_DIR}/logs/training/adv_feat_parallel \
            --output_dir ${BASE_DIR}/logs/shap/adv_feat_parallel
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Advanced feature selection with parallel processing completed."
    else
        echo "Advanced feature selection with parallel processing - Step $step completed."
    fi
}

# Function to run clustering approach with 3 clusters
run_cluster3_gcn() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Clustering Approach with 3 Clusters - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/cluster3_gcn \
            --use_advanced_feature_selection False \
            --use_clustering True \
            --n_clusters 3
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Clustering Approach with 3 Clusters - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster3_gcn \
            --output_dir ${BASE_DIR}/logs/training/cluster3_gcn \
            --model_type gcn \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Clustering Approach with 3 Clusters - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster3_gcn \
            --model_dir ${BASE_DIR}/logs/training/cluster3_gcn \
            --output_dir ${BASE_DIR}/logs/shap/cluster3_gcn
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Clustering approach with 3 clusters completed."
    else
        echo "Clustering approach with 3 clusters - Step $step completed."
    fi
}

# Function to run clustering approach with 5 clusters
run_cluster5_gcn() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Clustering Approach with 5 Clusters - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/cluster5_gcn \
            --use_advanced_feature_selection False \
            --use_clustering True \
            --n_clusters 5
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Clustering Approach with 5 Clusters - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster5_gcn \
            --output_dir ${BASE_DIR}/logs/training/cluster5_gcn \
            --model_type gcn \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Clustering Approach with 5 Clusters - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/cluster5_gcn \
            --model_dir ${BASE_DIR}/logs/training/cluster5_gcn \
            --output_dir ${BASE_DIR}/logs/shap/cluster5_gcn
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Clustering approach with 5 clusters completed."
    else
        echo "Clustering approach with 5 clusters - Step $step completed."
    fi
}

# Function to run combined approach
run_combined() {
    local step="$1"
    
    if [ -z "$step" ] || [ "$step" = "1" ]; then
        echo "=== Running Combined Approach - Step 1: Preprocessing data ==="
        python ${BASE_DIR}/scripts/01_preprocess_data.py \
            --data_file ${DATA_FILE} \
            --mi_file ${MI_FILE} \
            --output_dir ${BASE_DIR}/data/preprocessed/combined \
            --use_advanced_feature_selection True \
            --use_clustering True \
            --n_clusters 3 \
            --use_parallel True
        echo "Step 1 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "2" ]; then
        echo "=== Running Combined Approach - Step 2: Training model ==="
        python ${BASE_DIR}/scripts/02_train_model.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/combined \
            --output_dir ${BASE_DIR}/logs/training/combined \
            --model_type gcn \
            --hidden_dim ${HIDDEN_DIM} \
            --num_layers ${NUM_LAYERS}
        echo "Step 2 completed."
    fi
    
    if [ -z "$step" ] || [ "$step" = "3" ]; then
        echo "=== Running Combined Approach - Step 3: Analyzing bottlenecks ==="
        python ${BASE_DIR}/scripts/03_analyze_bottlenecks.py \
            --preprocessed_dir ${BASE_DIR}/data/preprocessed/combined \
            --model_dir ${BASE_DIR}/logs/training/combined \
            --output_dir ${BASE_DIR}/logs/shap/combined
        echo "Step 3 completed."
    fi
    
    if [ -z "$step" ]; then
        echo "Combined approach completed."
    else
        echo "Combined approach - Step $step completed."
    fi
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

# Function to run an individual script with custom arguments
run_individual_script() {
    local script_name="$1"
    shift
    local script_args="$@"
    
    echo "=== Running Individual Script: ${script_name}.py ==="
    
    # Check if script exists
    if [ ! -f "${BASE_DIR}/scripts/${script_name}.py" ]; then
        echo "Error: Script ${BASE_DIR}/scripts/${script_name}.py not found"
        exit 1
    fi
    
    # Run the script with the provided arguments
    echo "python ${BASE_DIR}/scripts/${script_name}.py ${script_args}"
    python ${BASE_DIR}/scripts/${script_name}.py ${script_args}
    
    echo "Script execution completed."
}

# Execute the specified command
case ${COMMAND} in
    split-data)
        run_split_data
        ;;
    process-splits)
        process_splits "${EXP_NAME}"
        ;;
    baseline-gcn)
        run_baseline_gcn "${STEP}"
        ;;
    baseline-gat)
        run_baseline_gat "${STEP}"
        ;;
    adv-feat-gcn)
        run_adv_feat_gcn "${STEP}"
        ;;
    adv-feat-parallel)
        run_adv_feat_parallel "${STEP}"
        ;;
    cluster3-gcn)
        run_cluster3_gcn "${STEP}"
        ;;
    cluster5-gcn)
        run_cluster5_gcn "${STEP}"
        ;;
    combined)
        run_combined "${STEP}"
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
    run-script)
        run_individual_script "${SCRIPT_NAME}" ${SCRIPT_ARGS}
        ;;
    *)
        echo "Error: Unknown command '${COMMAND}'"
        show_help
        exit 1
        ;;
esac

exit 0
