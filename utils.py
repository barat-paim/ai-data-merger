import pandas as pd
import sqlite3
from typing import Tuple, List
from sqlalchemy import create_engine, inspect
import tempfile
import os

def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to a temporary location and return the path."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def get_tables_from_db(db_path: str) -> List[str]:
    """Get list of tables from a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return [table[0] for table in tables]

def read_table_from_db(db_path: str, table_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read a table from SQLite database and return DataFrame and column types.
    
    Returns:
        Tuple of (DataFrame, list of column types)
    """
    conn = sqlite3.connect(db_path)
    
    # Get column information
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    column_types = [col[2] for col in columns_info]
    
    # Read data
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    return df, column_types

def validate_schema_compatibility(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Validate if two DataFrames have compatible schemas.
    
    Returns:
        bool: True if schemas are compatible, False otherwise
    """
    return (
        set(df1.columns) == set(df2.columns) and
        all(df1[col].dtype == df2[col].dtype for col in df1.columns)
    )

def export_to_sqlite(df: pd.DataFrame, output_path: str, table_name: str = 'merged_data'):
    """Export DataFrame to SQLite database."""
    try:
        # Create SQLite connection
        conn = sqlite3.connect(output_path)
        
        # Drop the existing table if it exists
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        
        # Write the new data
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
        # Verify the write
        written_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        if not written_df.equals(df):
            raise ValueError("Data verification failed")
            
        conn.commit()
        conn.close()
        
        return output_path
    except Exception as e:
        raise Exception(f"Error exporting to SQLite: {str(e)}")