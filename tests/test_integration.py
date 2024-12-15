import pytest

def test_complete_pipeline(merger, sample_df1, sample_df2):
    merged_df, stats = merger.merge_datasets(sample_df1, sample_df2)
    assert len(merged_df) == len(sample_df1) + len(sample_df2)
    assert stats['total_matches'] > 0
