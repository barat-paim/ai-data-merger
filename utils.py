import pandas as pd
import sqlite3
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to a temporary location and return the path."""
    try:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the file
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
            
        logger.info(f"Saved uploaded file to: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise Exception(f"Error saving uploaded file: {str(e)}")

def get_tables_from_db(db_path: str) -> list:
    """Get list of tables from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        logger.info(f"Found tables in database: {tables}")
        return tables
    except Exception as e:
        logger.error(f"Error reading tables from database: {str(e)}")
        raise Exception(f"Error reading database: {str(e)}")

def read_table_from_db(db_path: str, table_name: str) -> tuple:
    """Read a table from SQLite database and return DataFrame and column types."""
    try:
        conn = sqlite3.connect(db_path)
        
        # Get column types
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_info = cursor.fetchall()
        column_types = {col[1]: col[2] for col in columns_info}
        
        # Read data
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        logger.info(f"Read table {table_name} with shape {df.shape}")
        return df, column_types
    except Exception as e:
        logger.error(f"Error reading table from database: {str(e)}")
        raise Exception(f"Error reading table: {str(e)}")

def validate_schema_compatibility(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Check if two dataframes have compatible schemas for merging."""
    # Get column names
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    # Check for common columns
    common_cols = cols1.intersection(cols2)
    
    logger.info(f"Found {len(common_cols)} common columns between datasets")
    return len(common_cols) > 0

def export_to_sqlite(df: pd.DataFrame, output_path: str, table_name: str = 'merged_data') -> None:
    """Export DataFrame to SQLite database with data verification."""
    try:
        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Connect to SQLite database
        conn = sqlite3.connect(output_path)
        
        # Convert problematic data types
        df_clean = df.copy()
        for col in df_clean.columns:
            # Convert boolean columns to integer
            if df_clean[col].dtype == bool:
                df_clean[col] = df_clean[col].astype(int)
            # Convert any problematic object columns to string
            elif df_clean[col].dtype == object:
                df_clean[col] = df_clean[col].astype(str)
        
        # Export to SQLite
        df_clean.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Verify the data
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        db_count = cursor.fetchone()[0]
        
        if db_count != len(df):
            raise Exception("Data verification failed: Row count mismatch")
        
        # Close connection
        conn.close()
        
        logger.info(f"Successfully exported {len(df)} rows to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting to SQLite: {str(e)}")
        raise Exception(f"Error exporting to SQLite: {str(e)}")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names to be SQLite compatible."""
    df = df.copy()
    # Replace problematic characters and spaces
    df.columns = [col.strip().replace(' ', '_').replace('-', '_').lower() for col in df.columns]
    return df