# test_performance.py

import pytest
import pandas as pd
import numpy as np
import time
from merger import DatasetMerger

def generate_sample_data(size: int) -> pd.DataFrame:
    """Generate sample data for performance testing."""
    # Generate random data
    data = {
        'user_id': range(size),
        'name': [f"Person{i}" for i in range(size)],
        'age': np.random.randint(20, 80, size),
        'email': [f"person{i}@example.com" for i in range(size)],
        'score': np.random.uniform(0, 100, size),
        'status': np.random.choice(['active', 'inactive', 'pending'], size),
        'registration_date': pd.date_range(start='2020-01-01', periods=size).astype(str),
        'last_login': pd.date_range(start='2023-01-01', periods=size).astype(str)
    }
    return pd.DataFrame(data)

def generate_variant_data(df: pd.DataFrame, variation_factor: float = 0.2) -> pd.DataFrame:
    """Generate a variant dataset with some modifications."""
    size = len(df)
    num_variations = int(size * variation_factor)
    
    # Create a copy with slightly different column names
    data = {
        'userid': df['user_id'],
        'full_name': df['name'],
        'age_years': df['age'] + np.random.randint(-2, 3, size),
        'email_address': df['email'],
        'performance': df['score'] + np.random.uniform(-5, 5, size),
        'user_status': df['status'],
        'signup_date': df['registration_date'],
        'recent_login': df['last_login']
    }
    
    # Introduce some variations
    variant_df = pd.DataFrame(data)
    
    # Modify some values
    random_indices = np.random.choice(size, num_variations, replace=False)
    variant_df.loc[random_indices, 'full_name'] = [
        f"Modified_{name}" for name in variant_df.loc[random_indices, 'full_name']
    ]
    
    return variant_df

@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_processing_time(size):
    """Test processing time for different dataset sizes."""
    # Generate test data
    df1 = generate_sample_data(size)
    df2 = generate_variant_data(df1)
    
    # Measure processing time
    start_time = time.time()
    merger = DatasetMerger()
    merged_df, stats = merger.merge_datasets(df1, df2)
    processing_time = time.time() - start_time
    
    # Log performance metrics
    print(f"\nPerformance Metrics (size={size}):")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"Records per Second: {size/processing_time:.2f}")
    print(f"Matches Found: {stats['total_matches']}")
    
    # Assert performance requirements
    max_allowed_time = {
        100: 10,     # 10 seconds for 100 records
        1000: 60,    # 1 minute for 1000 records
        10000: 300   # 5 minutes for 10000 records
    }
    
    assert processing_time <= max_allowed_time[size], \
        f"Processing time {processing_time:.2f}s exceeded maximum allowed time {max_allowed_time[size]}s for size {size}"

@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_batch_size_impact(batch_size):
    """Test impact of different batch sizes on processing time."""
    size = 1000  # Fixed size for batch testing
    df1 = generate_sample_data(size)
    df2 = generate_variant_data(df1)
    
    start_time = time.time()
    merger = DatasetMerger()
    # Note: In a real implementation, you would need to modify the merger to accept batch_size
    merged_df, stats = merger.merge_datasets(df1, df2)
    processing_time = time.time() - start_time
    
    print(f"\nBatch Size Performance (batch_size={batch_size}, size={size}):")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"Records per Second: {size/processing_time:.2f}")

def test_memory_usage():
    """Test memory usage during processing."""
    import psutil
    import os
    
    size = 5000
    df1 = generate_sample_data(size)
    df2 = generate_variant_data(df1)
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Process data
    merger = DatasetMerger()
    merged_df, stats = merger.merge_datasets(df1, df2)
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    print(f"\nMemory Usage:")
    print(f"Initial Memory: {initial_memory:.2f} MB")
    print(f"Final Memory: {final_memory:.2f} MB")
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    # Assert reasonable memory usage (adjust thresholds as needed)
    assert memory_increase < 1024, f"Memory increase {memory_increase:.2f}MB exceeded 1GB threshold"