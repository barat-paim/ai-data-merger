import pytest
import pandas as pd
from merger import DatasetMerger

def test_complete_pipeline():
    """Test the entire preprocessing and merging pipeline."""
    # Create two sample databases
    df1 = pd.DataFrame({
        'user_id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', None],
        'age': [25, 30, None]
    })
    
    df2 = pd.DataFrame({
        'userid': [4, 5, 6],
        'full_name': ['Bob Wilson', 'Alice Brown', 'Charlie Davis'],
        'age_years': [35, None, 45]
    })
    
    # Process and merge - initialize with default parameters
    merger = DatasetMerger()
    merged_df, stats = merger.merge_datasets(df1, df2)
    
    # Verify results
    assert len(merged_df) == len(df1) + len(df2)
    assert stats['total_matches'] > 0
    assert not merged_df.isna().any().any()
