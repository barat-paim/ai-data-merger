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
    assert len(merged_df) == len(df1) + len(df2), "Merged dataset should contain all rows"
    assert stats['total_matches'] >= 0, "Total matches should be non-negative"
    
    # Check for NaN values in non-metadata columns
    non_metadata_cols = [col for col in merged_df.columns if col not in ['match_score', 'is_matched']]
    assert not merged_df[non_metadata_cols].isna().any().any(), "Found NaN values in non-metadata columns"
    
    # Verify metadata columns
    assert 'is_matched' in merged_df.columns, "is_matched column should be present"
    assert 'match_score' in merged_df.columns, "match_score column should be present"
    assert merged_df['is_matched'].dtype == bool, "is_matched should be boolean"
    
    # Verify schema matches
    assert len(stats['schema_matches']) > 0, "Should find some schema matches"
    assert 'matched_pairs' in stats, "matched_pairs should be in stats"
    assert 'unmatched_df1' in stats, "unmatched_df1 should be in stats"
    assert 'unmatched_df2' in stats, "unmatched_df2 should be in stats"
