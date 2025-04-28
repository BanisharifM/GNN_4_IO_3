#!/bin/bash

python -W ignore /u/mbanisharifdehkordi/Github/GNN_4_IO_3/scripts/02_train_model_single_checkpoint.py \
  --train_dir /u/mbanisharifdehkordi/Github/GNN_4_IO_3/data/preprocessed/baseline/mi0.2/train \
  --val_dir /u/mbanisharifdehkordi/Github/GNN_4_IO_3/data/preprocessed/baseline/mi0.2/val \
  --test_dir /u/mbanisharifdehkordi/Github/GNN_4_IO_3/data/preprocessed/baseline/mi0.2/test \
  --output_dir /u/mbanisharifdehkordi/Github/GNN_4_IO_3/logs/hyperopt/baseline/mi0.2/experiment_9/best_model \
  --hidden_dim 128 \
  --num_layers 3 \
  --model_type gat \
  --learning_rate 0.0001020917457159363 \
  --batch_size 16 \
  --dropout 0.31173467316200465 \
  --epochs 100 \
  --early_stopping_patience 10 \
  --use_split_dirs \
