import pytest
import pandas as pd
from preprocessor import DataPreprocessor
from merger import DatasetMerger

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
