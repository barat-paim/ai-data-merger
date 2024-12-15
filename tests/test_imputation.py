import pytest
import pandas as pd
import numpy as np

def test_numeric_imputation(preprocessor):
    df = pd.DataFrame({
        'value': [1, 2, None, 4, 5]
    })
    imputed_df = preprocessor.impute_missing_values(df)
    assert imputed_df['value'].isna().sum() == 0
