import pytest
import pandas as pd

def test_duplicate_removal(preprocessor):
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'name': ['John', 'John', 'Jane']
    })
    cleaned_df = preprocessor.clean_data(df)
    assert len(cleaned_df) == 2
