import pytest
import pandas as pd

def test_exact_column_matches(preprocessor, sample_df1):
    df2 = pd.DataFrame({'user_id': [3, 4], 'name': ['Bob', 'Alice']})
    matches = preprocessor.match_schemas(sample_df1, df2)
    assert matches == {'user_id': 'user_id', 'name': 'name'}
