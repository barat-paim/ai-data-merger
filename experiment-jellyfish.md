# Testing Guide for ML-Powered Dataset Merger

## Setup

```bash
# Create and activate virtual environment
python3 -m venv myenv39
source myenv39/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Run tests
python3 -m pytest tests/
```

## Test Categories

### 1. Schema Matching Tests

Test the jellyfish-based schema matching with various column scenarios:

```python
# test_schema_matching.py

def test_exact_column_matches():
    """Test matching of identical column names."""
    df1 = pd.DataFrame({'user_id': [1, 2], 'name': ['John', 'Jane']})
    df2 = pd.DataFrame({'user_id': [3, 4], 'name': ['Bob', 'Alice']})
    matches = preprocessor.match_schemas(df1, df2)
    assert matches == {'user_id': 'user_id', 'name': 'name'}

def test_similar_column_names():
    """Test matching of similar but not identical column names."""
    df1 = pd.DataFrame({'user_id': [1, 2], 'first_name': ['John', 'Jane']})
    df2 = pd.DataFrame({'userid': [3, 4], 'fname': ['Bob', 'Alice']})
    matches = preprocessor.match_schemas(df1, df2)
    assert matches['user_id'] == 'userid'
    assert matches['first_name'] == 'fname'

def test_content_based_matching():
    """Test matching based on column content similarity."""
    df1 = pd.DataFrame({'col1': ['John Doe', 'Jane Smith']})
    df2 = pd.DataFrame({'name': ['John Smith', 'Jane Doe']})
    matches = preprocessor.match_schemas(df1, df2)
    assert matches['col1'] == 'name'
```

### 2. Data Cleaning Tests

Test the data cleaning and standardization functionality:

```python
# test_data_cleaning.py

def test_duplicate_removal():
    """Test removal of duplicate rows."""
    df = pd.DataFrame({
        'id': [1, 1, 2],
        'name': ['John', 'John', 'Jane']
    })
    cleaned_df = preprocessor.clean_data(df)
    assert len(cleaned_df) == 2

def test_string_standardization():
    """Test standardization of string values."""
    df = pd.DataFrame({
        'name': [' John ', 'JANE', 'bob  ']
    })
    cleaned_df = preprocessor.clean_data(df)
    assert all(cleaned_df['name'] == ['john', 'jane', 'bob'])
```

### 3. Missing Value Imputation Tests

Test the imputation strategies for different data types:

```python
# test_imputation.py

def test_numeric_imputation():
    """Test median imputation for numeric columns."""
    df = pd.DataFrame({
        'value': [1, 2, None, 4, 5]
    })
    imputed_df = preprocessor.impute_missing_values(df)
    assert imputed_df['value'].isna().sum() == 0
    assert imputed_df['value'].median() == 3

def test_categorical_imputation():
    """Test mode imputation for categorical columns."""
    df = pd.DataFrame({
        'category': ['A', 'A', None, 'B', 'A']
    })
    imputed_df = preprocessor.impute_missing_values(df)
    assert imputed_df['category'].isna().sum() == 0
    assert imputed_df['category'].mode()[0] == 'A'
```

### 4. Integration Tests

Test the complete pipeline with real-world scenarios:

```python
# test_integration.py

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
    
    # Process and merge
    merger = DatasetMerger()
    merged_df, stats = merger.merge_datasets(df1, df2)
    
    # Verify results
    assert len(merged_df) == len(df1) + len(df2)
    assert stats['total_matches'] > 0
    assert not merged_df.isna().any().any()
```

## Performance Testing

### 1. Scalability Tests

Test performance with different dataset sizes:

```python
# test_performance.py

@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_processing_time(size):
    """Test processing time for different dataset sizes."""
    df1 = generate_sample_data(size)
    df2 = generate_sample_data(size)
    
    start_time = time.time()
    merger = DatasetMerger()
    merged_df, _ = merger.merge_datasets(df1, df2)
    processing_time = time.time() - start_time
    
    print(f"Size: {size}, Time: {processing_time:.2f}s")
```

### 2. Accuracy Tests

Measure matching accuracy with known ground truth:

```python
# test_accuracy.py

def test_matching_accuracy():
    """Test accuracy of schema matching and record matching."""
    # Create datasets with known matches
    df1, df2, true_matches = generate_datasets_with_ground_truth()
    
    merger = DatasetMerger()
    merged_df, stats = merger.merge_datasets(df1, df2)
    
    # Calculate accuracy metrics
    precision = calculate_precision(stats['matched_pairs'], true_matches)
    recall = calculate_recall(stats['matched_pairs'], true_matches)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    assert precision >= 0.8
    assert recall >= 0.8
    assert f1_score >= 0.8
```

## Experiment Scenarios

1. **Schema Variation Tests**
   - Test with identical schemas
   - Test with partially overlapping schemas
   - Test with completely different column names but similar content

2. **Data Quality Tests**
   - Test with clean data
   - Test with noisy data (typos, formatting inconsistencies)
   - Test with missing values
   - Test with duplicate records

3. **Scale Tests**
   - Small datasets (<1000 rows)
   - Medium datasets (1000-10000 rows)
   - Large datasets (>10000 rows)

4. **Content Type Tests**
   - Numeric data only
   - Text data only
   - Mixed data types
   - Special characters and Unicode

5. **Edge Cases**
   - Empty datasets
   - Single column datasets
   - All columns different
   - All values null

## Success Metrics

1. **Schema Matching Accuracy**
   - Precision: >90% for exact matches
   - Recall: >80% for similar column names
   - F1 Score: >85% overall

2. **Data Quality**
   - 100% duplicate removal
   - 100% missing value imputation
   - Consistent string formatting

3. **Performance Targets**
   - Processing time < 1s for small datasets
   - Linear scaling with dataset size
   - Memory usage within 2x dataset size

4. **Integration Success**
   - All components working together
   - No data loss during processing
   - Correct handling of all data types

## Running the Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_schema_matching.py

# Run with coverage report
pytest --cov=src tests/

# Run performance tests
pytest tests/test_performance.py -v
```

## Reporting

Generate test reports using:

```bash
pytest --html=report.html
```

This will create a detailed HTML report including:
- Test results
- Performance metrics
- Coverage information
- Error logs and tracebacks
