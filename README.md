# Dataset Merger with ML

This project provides a machine learning-based solution for merging two datasets with the same schema but different rows. It uses Hugging Face's sentence transformers for semantic similarity matching and provides a Streamlit interface for easy interaction.

## Features

- Upload and process SQL database files
- ML-based dataset merging using sentence transformers
- Interactive Streamlit interface for visualization
- Configurable similarity thresholds
- Export merged results

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## How it works

1. The system uses sentence transformers from Hugging Face to encode row data
2. Semantic similarity is computed between rows from both datasets
3. Similar rows are merged based on configurable threshold
4. Results are presented in an interactive interface

## Project Structure

- `app.py`: Main Streamlit application
- `merger.py`: Core dataset merging logic
- `utils.py`: Utility functions for data processing
- `requirements.txt`: Project dependencies


# format of the database
1. table name: typing_stats
2. tables are stored in the format of: table_name.sql
3. it can be converted to csv by using: sqlite3 table_name.sql -csv > table_name.csv