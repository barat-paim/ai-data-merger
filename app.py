import streamlit as st
import pandas as pd
from merger import DatasetMerger
from utils import (
    save_uploaded_file,
    get_tables_from_db,
    read_table_from_db,
    validate_schema_compatibility,
    export_to_sqlite
)
import os
import tempfile
import logging
import sys
import datetime
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters and handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

st.set_page_config(page_title="Dataset Manager", layout="wide")

# Initialize session state for database storage
if 'merger' not in st.session_state:
    st.session_state.merger = DatasetMerger()
if 'current_df' not in st.session_state:
    st.session_state.current_df = None

st.title("ML-Powered Dataset Merger")
st.write("""
This application helps you merge two datasets using machine learning.
Upload your SQLite databases, select tables, and the app will intelligently merge the data.
""")

# File upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("First Database")
    logger.info("Waiting for first database upload...")
    db1_file = st.file_uploader("Upload first SQLite database", type=['db', 'sqlite', 'sqlite3'], key="db1")
    
    if db1_file:
        logger.info(f"First database uploaded: {db1_file.name}")
        db1_path = save_uploaded_file(db1_file)
        logger.debug(f"Saved first database to: {db1_path}")
        
        tables1 = get_tables_from_db(db1_path)
        logger.debug(f"Available tables in first database: {tables1}")
        
        selected_table1 = st.selectbox("Select table from first database", tables1)
        if selected_table1:
            logger.info(f"Selected table from first database: {selected_table1}")
            df1, types1 = read_table_from_db(db1_path, selected_table1)
            logger.debug(f"First dataset shape: {df1.shape}")
            logger.debug(f"First dataset columns: {df1.columns.tolist()}")
            # Display shape metrics
            shape_col1, shape_col2 = st.columns(2)
            shape_col1.metric("Rows", df1.shape[0])
            shape_col2.metric("Columns", df1.shape[1])
            st.write("Preview of first dataset:")
            st.dataframe(df1.head(10), use_container_width=True)
            st.write("Column types:")
            st.json(dict(zip(df1.columns, types1)))
            
with col2:
    st.subheader("Second Database")
    logger.info("Waiting for second database upload...")
    db2_file = st.file_uploader("Upload second SQLite database", type=['db', 'sqlite', 'sqlite3'], key="db2")
    
    if db2_file:
        logger.info(f"Second database uploaded: {db2_file.name}")
        db2_path = save_uploaded_file(db2_file)
        logger.debug(f"Saved second database to: {db2_path}")
        
        tables2 = get_tables_from_db(db2_path)
        logger.debug(f"Available tables in second database: {tables2}")
        
        selected_table2 = st.selectbox("Select table from second database", tables2)
        if selected_table2:
            logger.info(f"Selected table from second database: {selected_table2}")
            df2, types2 = read_table_from_db(db2_path, selected_table2)
            logger.debug(f"Second dataset shape: {df2.shape}")
            logger.debug(f"Second dataset columns: {df2.columns.tolist()}")
            # Display shape metrics
            shape_col1, shape_col2 = st.columns(2)
            shape_col1.metric("Rows", df2.shape[0])
            shape_col2.metric("Columns", df2.shape[1])
            st.write("Preview of second dataset:")
            st.dataframe(df2.head(10), use_container_width=True)
            st.write("Column types:")
            st.json(dict(zip(df2.columns, types2)))

# Merging parameters
st.subheader("Merge Settings")
similarity_threshold = st.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    help="Minimum similarity score required to consider two rows as matching"
)

# Merge button and process
if st.button("Merge Datasets"):
    logger.info("Merge button clicked")
    if 'df1' not in locals() or 'df2' not in locals():
        logger.error("Attempted merge without both databases loaded")
        st.error("Please upload both databases and select tables first.")
    else:
        logger.info("Checking schema compatibility...")
        if not validate_schema_compatibility(df1, df2):
            logger.error("No common columns found between datasets")
            st.error("No common columns found between the datasets. The datasets must share at least one column.")
        else:
            with st.spinner("Merging datasets..."):
                # Show common and unique columns
                common_cols = set(df1.columns).intersection(set(df2.columns))
                unique_cols_df1 = set(df1.columns) - set(df2.columns)
                unique_cols_df2 = set(df2.columns) - set(df1.columns)
                
                logger.info(f"Common columns: {common_cols}")
                logger.info(f"Unique columns in df1: {unique_cols_df1}")
                logger.info(f"Unique columns in df2: {unique_cols_df2}")
                
                merged_df, stats = st.session_state.merger.merge_datasets(
                    df1, df2, similarity_threshold=similarity_threshold
                )
                
                # Display shape comparison
                st.subheader("Dataset Shapes")
                shape_cols = st.columns(3)
                with shape_cols[0]:
                    st.metric("Dataset 1", f"{df1.shape[0]} × {df1.shape[1]}", 
                             delta=f"Rows: {df1.shape[0]}")
                with shape_cols[1]:
                    st.metric("Dataset 2", f"{df2.shape[0]} × {df2.shape[1]}", 
                             delta=f"Rows: {df2.shape[0]}")
                with shape_cols[2]:
                    st.metric("Merged Dataset", f"{merged_df.shape[0]} × {merged_df.shape[1]}", 
                             delta=f"Rows: {merged_df.shape[0]}")
                
                # Display merge statistics
                st.subheader("Merge Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Matches", stats['total_matches'])
                col2.metric("Unmatched from DB1", stats['unmatched_df1'])
                col3.metric("Unmatched from DB2", stats['unmatched_df2'])
                col4.metric("Total Rows", stats['total_rows'])
                
                # Display merged dataset with detailed information
                st.subheader("Merged Dataset Preview")
                st.dataframe(merged_df.head(20), use_container_width=True)
                st.write(f"Total rows in merged dataset: {len(merged_df)}")
                
                # Show sample of matched rows
                if stats['total_matches'] > 0:
                    st.subheader("Sample of Matched Rows")
                    matched_indices = [pair[0] for pair in stats.get('matched_pairs', [])][:5]
                    if matched_indices:
                        st.write("From Dataset 1:")
                        st.dataframe(df1.iloc[matched_indices], use_container_width=True)
                        st.write("Corresponding rows from Dataset 2:")
                        st.dataframe(df2.iloc[matched_indices], use_container_width=True)
                
                # Export options
                st.subheader("Export Options")
                if st.button("Export as SQLite"):
                    logger.info("Export button clicked")
                    try:
                        # Create output directory if it doesn't exist
                        output_dir = Path("merged_databases")
                        output_dir.mkdir(exist_ok=True)
                        
                        # Generate unique filename
                        filename = generate_unique_filename()
                        output_path = output_dir / filename
                        
                        logger.info(f"Saving merged database to: {output_path}")
                        export_to_sqlite(merged_df, str(output_path))
                        logger.info("Database saved successfully")
                        
                        # Provide both download and local path information
                        st.success(f"Database saved locally at: {output_path}")
                        
                        with open(output_path, 'rb') as f:
                            logger.debug("Preparing download button")
                            st.download_button(
                                label="Download merged database",
                                data=f,
                                file_name=filename,
                                mime="application/x-sqlite3"
                            )
                        
                        # Display additional file information
                        file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
                        logger.info(f"File size: {file_size:.2f} MB")
                        st.info(f"""
                        File Details:
                        - Size: {file_size:.2f} MB
                        - Location: {output_path.absolute()}
                        - Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        """)
                        
                    except Exception as e:
                        logger.error(f"Error during export: {str(e)}", exc_info=True)
                        st.error(f"Error during export: {str(e)}")

def generate_unique_filename(prefix: str = "merged_db") -> str:
    """Generate a unique filename with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.db"