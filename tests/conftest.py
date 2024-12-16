import sys
import os
import pytest
import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessor import DataPreprocessor
from merger import DatasetMerger
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance for testing."""
    return DataPreprocessor()

@pytest.fixture
def merger():
    """Create a merger instance for testing."""
    return DatasetMerger()

@pytest.fixture
def sample_df1():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'user_id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', None],
        'age': [25, 30, None]
    })

@pytest.fixture
def sample_df2():
    """Create another sample dataframe for testing."""
    return pd.DataFrame({
        'userid': [4, 5, 6],
        'full_name': ['Bob Wilson', 'Alice Brown', 'Charlie Davis'],
        'age_years': [35, None, 45]
    })
