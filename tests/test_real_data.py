import pytest
import pandas as pd
import numpy as np
from merger import DatasetMerger
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_data_merge():
    """Test merging real customer data with different schemas and sizes."""
    # Load the datasets
    df1 = pd.read_csv('data/customers1.csv')
    df2 = pd.read_csv('data/customers2.csv')
    
    # Print initial information
    logger.info(f"\nDataset 1 shape: {df1.shape}")
    logger.info(f"Dataset 1 columns: {df1.columns.tolist()}")
    logger.info(f"\nDataset 2 shape: {df2.shape}")
    logger.info(f"Dataset 2 columns: {df2.columns.tolist()}")
    
    # Initialize merger
    merger = DatasetMerger()
    
    # Perform merge
    merged_df, stats = merger.merge_datasets(df1, df2)
    
    # Print merge statistics
    logger.info("\nMerge Statistics:")
    logger.info(f"Total matches found: {stats['total_matches']}")
    logger.info(f"Schema matches: {stats['schema_matches']}")
    logger.info(f"Unmatched in df1: {stats['unmatched_df1']}")
    logger.info(f"Unmatched in df2: {stats['unmatched_df2']}")
    
    # Verify the results
    logger.info(f"\nMerged dataset shape: {merged_df.shape}")
    logger.info(f"Merged columns: {merged_df.columns.tolist()}")
    
    # Expected matches (we know these records exist in both datasets)
    expected_matches = {'1001', '1003', '1005', '1008', '1009'}
    
    # Calculate actual matches
    actual_matches = set()
    for idx1, idx2, _ in stats['matched_pairs']:
        id1 = str(df1.iloc[idx1]['customer_id'])
        id2 = str(df2.iloc[idx2]['cust_id'])
        if id1 == id2:
            actual_matches.add(id1)
    
    # Assertions
    assert len(actual_matches) >= len(expected_matches), \
        f"Found only {len(actual_matches)} matches, expected at least {len(expected_matches)}"
    
    assert actual_matches.issuperset(expected_matches), \
        f"Missing expected matches: {expected_matches - actual_matches}"
    
    # Verify schema matching
    expected_schema_matches = {
        'customer_id': 'cust_id',
        'email': 'email',
        'city': 'city',
        'total_purchases': 'purchases'
    }
    
    actual_schema_matches = stats['schema_matches']
    for key, value in expected_schema_matches.items():
        assert key in actual_schema_matches and actual_schema_matches[key] == value, \
            f"Expected schema match {key} -> {value} not found"
    
    # Verify no data loss
    assert len(merged_df) == len(df1) + len(df2), \
        "Merged dataset should contain all rows from both datasets"
    
    # Verify all required columns are present
    required_columns = set(df1.columns) | set(df2.columns)
    merged_columns = set(merged_df.columns)
    missing_columns = required_columns - merged_columns
    logger.info(f"Required columns: {required_columns}")
    logger.info(f"Merged columns: {merged_columns}")
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
    assert not missing_columns, f"Missing columns: {missing_columns}"
    
    # Verify no NaN values in key columns
    key_columns = ['customer_id', 'email', 'city']
    for col in key_columns:
        if col in merged_df.columns:
            assert not merged_df[col].isna().any(), \
                f"Found NaN values in key column: {col}"
    
    logger.info("\nAll assertions passed successfully!")

if __name__ == '__main__':
    test_real_data_merge() 