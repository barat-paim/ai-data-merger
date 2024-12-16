import pytest
import pandas as pd
import numpy as np
from merger import DatasetMerger
from preprocessor import DataPreprocessor

def generate_datasets_with_ground_truth(size=100, match_ratio=0.7):
    """Generate test datasets with known ground truth matches."""
    # Generate base data for matches
    num_matches = int(size * match_ratio)
    
    # Create matched records
    matched_ids = range(num_matches)
    matched_names = [f"Person{i}" for i in matched_ids]
    matched_ages = np.random.randint(20, 80, num_matches)
    
    # Create df1 with some variations
    df1_data = {
        'user_id': matched_ids,
        'name': matched_names,
        'age': matched_ages
    }
    df1 = pd.DataFrame(df1_data)
    
    # Create df2 with variations but maintaining ground truth
    df2_data = {
        'userid': matched_ids,  # Same IDs but different column name
        'full_name': [name.upper() for name in matched_names],  # Names in uppercase
        'age_years': matched_ages + np.random.randint(-2, 3, num_matches)  # Slightly different ages
    }
    df2 = pd.DataFrame(df2_data)
    
    # Add unmatched records to both dataframes
    unmatched_size = size - num_matches
    
    # Add unmatched records to df1
    df1_unmatched = pd.DataFrame({
        'user_id': range(num_matches, num_matches + unmatched_size),
        'name': [f"Unique1_{i}" for i in range(unmatched_size)],
        'age': np.random.randint(20, 80, unmatched_size)
    })
    df1 = pd.concat([df1, df1_unmatched], ignore_index=True)
    
    # Add unmatched records to df2
    df2_unmatched = pd.DataFrame({
        'userid': range(num_matches + unmatched_size, num_matches + 2*unmatched_size),
        'full_name': [f"Unique2_{i}" for i in range(unmatched_size)],
        'age_years': np.random.randint(20, 80, unmatched_size)
    })
    df2 = pd.concat([df2, df2_unmatched], ignore_index=True)
    
    # Create ground truth matches
    true_matches = [(i, i) for i in range(num_matches)]
    
    return df1, df2, true_matches

def calculate_precision(matched_pairs, true_matches):
    """Calculate precision of the matching."""
    if not matched_pairs:
        return 0.0
    
    # Convert matched_pairs to set of tuples (idx1, idx2)
    predicted_matches = {(idx1, idx2) for idx1, idx2, _ in matched_pairs}
    true_matches_set = set(true_matches)
    
    # Calculate true positives and precision
    true_positives = len(predicted_matches.intersection(true_matches_set))
    precision = true_positives / len(predicted_matches)
    
    return precision

def calculate_recall(matched_pairs, true_matches):
    """Calculate recall of the matching."""
    if not true_matches:
        return 0.0
    
    # Convert matched_pairs to set of tuples (idx1, idx2)
    predicted_matches = {(idx1, idx2) for idx1, idx2, _ in matched_pairs}
    true_matches_set = set(true_matches)
    
    # Calculate true positives and recall
    true_positives = len(predicted_matches.intersection(true_matches_set))
    recall = true_positives / len(true_matches_set)
    
    return recall

@pytest.mark.parametrize("size,match_ratio", [
    (100, 0.7),
    (200, 0.8),
    (50, 0.6)
])
def test_matching_accuracy(size, match_ratio):
    """Test accuracy of schema matching and record matching."""
    # Create datasets with known matches
    df1, df2, true_matches = generate_datasets_with_ground_truth(size, match_ratio)
    
    merger = DatasetMerger()
    merged_df, stats = merger.merge_datasets(df1, df2)
    
    # Calculate accuracy metrics
    precision = calculate_precision(stats['matched_pairs'], true_matches)
    recall = calculate_recall(stats['matched_pairs'], true_matches)
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # Log results
    print(f"\nAccuracy Metrics (size={size}, match_ratio={match_ratio}):")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    
    # Assert minimum performance requirements
    assert precision >= 0.8, f"Precision {precision:.2f} below threshold 0.8"
    assert recall >= 0.8, f"Recall {recall:.2f} below threshold 0.8"
    assert f1_score >= 0.8, f"F1 Score {f1_score:.2f} below threshold 0.8"

def test_schema_matching_accuracy():
    """Test accuracy of schema matching specifically."""
    # Create test datasets with known schema matches
    df1 = pd.DataFrame({
        'user_id': [1, 2],
        'first_name': ['John', 'Jane'],
        'age_in_years': [25, 30]
    })
    
    df2 = pd.DataFrame({
        'userid': [3, 4],
        'fname': ['Bob', 'Alice'],
        'age': [35, 40]
    })
    
    merger = DatasetMerger()
    _, stats = merger.merge_datasets(df1, df2)
    
    # Expected schema matches
    expected_matches = {
        'user_id': 'userid',
        'first_name': 'fname',
        'age_in_years': 'age'
    }
    
    # Compare actual schema matches with expected
    actual_matches = stats['schema_matches']  # Access schema_matches directly
    assert len(actual_matches) >= len(expected_matches), \
        f"Found only {len(actual_matches)} schema matches, expected at least {len(expected_matches)}"
    
    # Verify that the expected matches are present
    for key, value in expected_matches.items():
        assert key in actual_matches, f"Expected match for {key} not found"
        assert actual_matches[key] == value, \
            f"Expected {key} to match with {value}, but got {actual_matches[key]}"
