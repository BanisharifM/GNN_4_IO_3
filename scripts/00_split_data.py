#!/usr/bin/env python
# Script to create permanent train/validation/test splits

import os
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import time
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_data(
    data_file,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    sample_size=None,
    hash_check=True
):
    """
    Split data into train, validation, and test sets and save as separate files.
    
    Args:
        data_file (str): Path to the data CSV file
        output_dir (str): Directory to save split data files
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        random_seed (int): Random seed for reproducibility
        sample_size (int, optional): Number of samples to use (for testing)
        hash_check (bool): Whether to add hash check to ensure data consistency
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {data_file}")
    start_time = time.time()
    data_df = pd.read_csv(data_file)
    load_time = time.time() - start_time
    logger.info(f"Data loaded in {load_time:.2f} seconds")
    
    # Sample data if sample_size is provided
    if sample_size is not None and sample_size < len(data_df):
        logger.info(f"Sampling {sample_size} rows from dataset")
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        data_df = data_df.sample(n=sample_size, random_state=random_seed)
    
    # Calculate data hash for consistency check
    if hash_check:
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data_df).values).hexdigest()
        logger.info(f"Data hash: {data_hash}")
        
        # Save hash to file
        with open(os.path.join(output_dir, 'data_hash.txt'), 'w') as f:
            f.write(data_hash)
    
    # Shuffle data
    logger.info("Shuffling data")
    shuffled_df = data_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate split sizes
    total_size = len(shuffled_df)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(f"Splitting data: {train_size} train, {val_size} validation, {test_size} test")
    
    # Split data
    train_df = shuffled_df[:train_size]
    val_df = shuffled_df[train_size:train_size + val_size]
    test_df = shuffled_df[train_size + val_size:]
    
    # Save split indices
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(output_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    
    # Save split data
    logger.info("Saving split data files")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save split information
    split_info = {
        'data_file': data_file,
        'total_size': total_size,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if hash_check:
        split_info['data_hash'] = data_hash
    
    # Save as JSON
    import json
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
    
    logger.info(f"Data splitting completed. Files saved to {output_dir}")
    
    return split_info

def verify_split(data_file, split_dir):
    """
    Verify that the split data is consistent with the original data.
    
    Args:
        data_file (str): Path to the original data CSV file
        split_dir (str): Directory with split data files
    """
    logger.info(f"Verifying split data consistency")
    
    # Load original data
    original_df = pd.read_csv(data_file)
    
    # Load split data
    train_df = pd.read_csv(os.path.join(split_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(split_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, 'test.csv'))
    
    # Check sizes
    original_size = len(original_df)
    split_size = len(train_df) + len(val_df) + len(test_df)
    
    if 'data_hash.txt' in os.listdir(split_dir):
        with open(os.path.join(split_dir, 'data_hash.txt'), 'r') as f:
            saved_hash = f.read().strip()
        
        # Calculate current hash
        current_hash = hashlib.md5(pd.util.hash_pandas_object(original_df).values).hexdigest()
        
        if saved_hash != current_hash:
            logger.warning(f"Data hash mismatch! Original data may have changed since splitting.")
            logger.warning(f"Saved hash: {saved_hash}")
            logger.warning(f"Current hash: {current_hash}")
    
    # Check if split size matches original size (allowing for sampling)
    with open(os.path.join(split_dir, 'split_info.json'), 'r') as f:
        split_info = json.load(f)
    
    expected_size = split_info['total_size']
    
    if split_size != expected_size:
        logger.error(f"Split size mismatch! Expected {expected_size}, got {split_size}")
        return False
    
    logger.info(f"Split verification completed.")
    logger.info(f"Original size: {original_size}, Split size: {split_size}")
    
    if original_size != split_size:
        logger.info(f"Note: Split size differs from original size due to sampling.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, validation, and test sets")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split data files")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of test data")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use (for testing)")
    parser.add_argument("--verify", action="store_true", help="Verify split data consistency")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_split(args.data_file, args.output_dir)
    else:
        split_info = split_data(
            args.data_file,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.random_seed,
            args.sample_size
        )
        
        print("\nData Splitting Results:")
        print(f"Total size: {split_info['total_size']}")
        print(f"Train size: {split_info['train_size']} ({split_info['train_ratio'] * 100:.1f}%)")
        print(f"Validation size: {split_info['val_size']} ({split_info['val_ratio'] * 100:.1f}%)")
        print(f"Test size: {split_info['test_size']} ({split_info['test_ratio'] * 100:.1f}%)")
        print(f"Files saved to: {args.output_dir}")
