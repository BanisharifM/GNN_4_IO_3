(gnn-env) (base) mbanisharifdehkordi@dt-login01: /u/mbanisharifdehkordi/Github/GNN_4_IO_3 git:(main) ✗ 
➜   ./run/preprocess_test.sh 
Preprocessing test data from data/split_data/sample_total/test.csv...
2025-04-08 23:37:13 - INFO - Preprocessing data from data/split_data/sample_total/test.csv
2025-04-08 23:37:13 - INFO - Using mutual information from data/mutual_information2.csv
2025-04-08 23:37:13 - INFO - Loading data...

2025-04-08 23:37:59 - INFO - Constructing graph with MI threshold: 0.3259
2025-04-08 23:37:59 - INFO - Loading mutual information from data/mutual_information2.csv
2025-04-08 23:37:59 - INFO - Constructing graph with MI threshold: 0.3259
2025-04-08 23:37:59 - INFO - Created graph with 44 nodes and 84 edges
2025-04-08 23:37:59 - INFO - Exported graph structure to data/preprocessed/baseline_gcn/experiment_3/test/graph_structure.json
2025-04-08 23:37:59 - INFO - Exported graph edges to data/preprocessed/baseline_gcn/experiment_3/test/graph_edges.csv
2025-04-08 23:37:59 - INFO - Exported graph nodes to data/preprocessed/baseline_gcn/experiment_3/test/graph_nodes.csv
2025-04-08 23:37:59 - INFO - Exported graph edges tensor to data/preprocessed/baseline_gcn/experiment_3/test/edge_index.pt
2025-04-08 23:37:59 - INFO - Exported graph edge attributes tensor to data/preprocessed/baseline_gcn/experiment_3/test/edge_attr.pt
2025-04-08 23:37:59 - INFO - Saved counter mapping to data/preprocessed/baseline_gcn/experiment_3/test/counter_mapping.csv
2025-04-08 23:37:59 - INFO - Extracting node features...
2025-04-08 23:40:45 - INFO - Saved node features tensor to data/preprocessed/baseline_gcn/experiment_3/test/node_features.pt
2025-04-08 23:40:45 - INFO - Saved targets tensor to data/preprocessed/baseline_gcn/experiment_3/test/targets.pt
2025-04-08 23:41:14 - INFO - Saving preprocessed data to data/preprocessed/baseline_gcn/experiment_3/test
2025-04-08 23:42:46 - INFO - Preprocessing completed. Data saved to data/preprocessed/baseline_gcn/experiment_3/test
2025-04-08 23:42:46 - INFO - dataset_size: 997084
2025-04-08 23:42:46 - INFO - num_nodes: 44
2025-04-08 23:42:46 - INFO - num_edges: 168
2025-04-08 23:42:46 - INFO - test_size: 997084

Preprocessing Statistics:
dataset_size: 997084
num_nodes: 44
num_edges: 168
test_size: 997084
Preprocessing completed. Results saved to data/preprocessed/baseline_gcn/experiment_3/test/


(gnn-env) (base) mbanisharifdehkordi@dt-login01: /u/mbanisharifdehkordi/Github/GNN_4_IO_3 git:(main) ✗ 
➜   ./run/preprocess_val.sh 
Preprocessing val data from data/split_data/sample_total/val.csv...
2025-04-08 23:43:17 - INFO - Preprocessing data from data/split_data/sample_total/val.csv
2025-04-08 23:43:17 - INFO - Using mutual information from data/mutual_information2.csv
2025-04-08 23:43:17 - INFO - Loading data...
2025-04-08 23:43:56 - INFO - Constructing graph with MI threshold: 0.3259
2025-04-08 23:43:56 - INFO - Loading mutual information from data/mutual_information2.csv
2025-04-08 23:43:56 - INFO - Constructing graph with MI threshold: 0.3259
2025-04-08 23:43:56 - INFO - Created graph with 44 nodes and 84 edges
2025-04-08 23:43:56 - INFO - Exported graph structure to data/preprocessed/baseline_gcn/experiment_3/val/graph_structure.json
2025-04-08 23:43:56 - INFO - Exported graph edges to data/preprocessed/baseline_gcn/experiment_3/val/graph_edges.csv
2025-04-08 23:43:56 - INFO - Exported graph nodes to data/preprocessed/baseline_gcn/experiment_3/val/graph_nodes.csv
2025-04-08 23:43:56 - INFO - Exported graph edges tensor to data/preprocessed/baseline_gcn/experiment_3/val/edge_index.pt
2025-04-08 23:43:56 - INFO - Exported graph edge attributes tensor to data/preprocessed/baseline_gcn/experiment_3/val/edge_attr.pt
2025-04-08 23:43:56 - INFO - Saved counter mapping to data/preprocessed/baseline_gcn/experiment_3/val/counter_mapping.csv
2025-04-08 23:43:56 - INFO - Extracting node features...
2025-04-08 23:46:46 - INFO - Saved node features tensor to data/preprocessed/baseline_gcn/experiment_3/val/node_features.pt
2025-04-08 23:46:46 - INFO - Saved targets tensor to data/preprocessed/baseline_gcn/experiment_3/val/targets.pt
2025-04-08 23:47:19 - INFO - Saving preprocessed data to data/preprocessed/baseline_gcn/experiment_3/val
2025-04-08 23:48:52 - INFO - Preprocessing completed. Data saved to data/preprocessed/baseline_gcn/experiment_3/val
2025-04-08 23:48:52 - INFO - dataset_size: 997082
2025-04-08 23:48:52 - INFO - num_nodes: 44
2025-04-08 23:48:52 - INFO - num_edges: 168
2025-04-08 23:48:52 - INFO - val_size: 997082

Preprocessing Statistics:
dataset_size: 997082
num_nodes: 44
num_edges: 168
val_size: 997082
Preprocessing completed. Results saved to data/preprocessed/baseline_gcn/experiment_3/val/

(gnn-env) (base) mbanisharifdehkordi@dt-login01: /u/mbanisharifdehkordi/Github/GNN_4_IO_3 git:(main) ✗ 
➜   ./run/preprocess_train.sh 
Preprocessing train data from data/split_data/sample_total/train.csv...
2025-04-08 23:49:36 - INFO - Preprocessing data from data/split_data/sample_total/train.csv
2025-04-08 23:49:36 - INFO - Using mutual information from data/mutual_information2.csv
2025-04-08 23:49:36 - INFO - Loading data...
2025-04-08 23:52:15 - INFO - Constructing graph with MI threshold: 0.3259
2025-04-08 23:52:15 - INFO - Loading mutual information from data/mutual_information2.csv
2025-04-08 23:52:15 - INFO - Constructing graph with MI threshold: 0.3259
2025-04-08 23:52:15 - INFO - Created graph with 44 nodes and 84 edges
2025-04-08 23:52:15 - INFO - Exported graph structure to data/preprocessed/baseline_gcn/experiment_3/train/graph_structure.json
2025-04-08 23:52:15 - INFO - Exported graph edges to data/preprocessed/baseline_gcn/experiment_3/train/graph_edges.csv
2025-04-08 23:52:15 - INFO - Exported graph nodes to data/preprocessed/baseline_gcn/experiment_3/train/graph_nodes.csv
2025-04-08 23:52:15 - INFO - Exported graph edges tensor to data/preprocessed/baseline_gcn/experiment_3/train/edge_index.pt
2025-04-08 23:52:15 - INFO - Exported graph edge attributes tensor to data/preprocessed/baseline_gcn/experiment_3/train/edge_attr.pt
2025-04-08 23:52:15 - INFO - Saved counter mapping to data/preprocessed/baseline_gcn/experiment_3/train/counter_mapping.csv
2025-04-08 23:52:15 - INFO - Extracting node features...
2025-04-09 00:05:46 - INFO - Saved node features tensor to data/preprocessed/baseline_gcn/experiment_3/train/node_features.pt
2025-04-09 00:05:46 - INFO - Saved targets tensor to data/preprocessed/baseline_gcn/experiment_3/train/targets.pt
2025-04-09 00:08:16 - INFO - Saving preprocessed data to data/preprocessed/baseline_gcn/experiment_3/train
2025-04-09 00:15:49 - INFO - Preprocessing completed. Data saved to data/preprocessed/baseline_gcn/experiment_3/train
2025-04-09 00:15:49 - INFO - dataset_size: 4653053
2025-04-09 00:15:49 - INFO - num_nodes: 44
2025-04-09 00:15:49 - INFO - num_edges: 168
2025-04-09 00:15:49 - INFO - train_size: 4653053

Preprocessing Statistics:
dataset_size: 4653053
num_nodes: 44
num_edges: 168
train_size: 4653053
Preprocessing completed. Results saved to data/preprocessed/baseline_gcn/experiment_3/train/

