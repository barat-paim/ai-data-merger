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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Dataset Manager", layout="wide")

# Initialize session state for database storage
if 'merger' not in st.session_state:
    st.session_state.merger = DatasetMerger()
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'current_db_path' not in st.session_state:
    st.session_state.current_db_path = None
if 'current_table' not in st.session_state:
    st.session_state.current_table = None

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Dataset Merger", "Column Manager"])

if page == "Dataset Merger":
    st.title("ML-Powered Dataset Merger")
    st.write("""
    This application helps you merge two datasets with the same schema using machine learning.
    Upload your SQLite databases, select tables, and the app will intelligently merge the data.
    """)

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First Database")
        db1_file = st.file_uploader("Upload first SQLite database", type=['db', 'sqlite', 'sqlite3'], key="db1")
        
        if db1_file:
            db1_path = save_uploaded_file(db1_file)
            tables1 = get_tables_from_db(db1_path)
            selected_table1 = st.selectbox("Select table from first database", tables1)
            if selected_table1:
                df1, types1 = read_table_from_db(db1_path, selected_table1)
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
        db2_file = st.file_uploader("Upload second SQLite database", type=['db', 'sqlite', 'sqlite3'], key="db2")
        
        if db2_file:
            db2_path = save_uploaded_file(db2_file)
            tables2 = get_tables_from_db(db2_path)
            selected_table2 = st.selectbox("Select table from second database", tables2)
            if selected_table2:
                df2, types2 = read_table_from_db(db2_path, selected_table2)
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
        if 'df1' not in locals() or 'df2' not in locals():
            st.error("Please upload both databases and select tables first.")
        else:
            if not validate_schema_compatibility(df1, df2):
                st.error("The selected tables have incompatible schemas. Please ensure they have the same columns and data types.")
            else:
                with st.spinner("Merging datasets..."):
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
                        temp_dir = tempfile.mkdtemp()
                        output_path = os.path.join(temp_dir, "merged_database.db")
                        export_to_sqlite(merged_df, output_path)
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download merged database",
                                data=f,
                                file_name="merged_database.db",
                                mime="application/x-sqlite3"
                            )

elif page == "Column Manager":
    logger.info("Entering Column Manager page")
    st.title("Column Manager")
    st.write("""
    Modify the structure of your database tables by adding or removing columns.
    Upload a database and select a table to begin.
    """)

    # Database upload and table selection
    uploaded_file = st.file_uploader("Upload SQLite database", type=['db', 'sqlite', 'sqlite3'], key="db_column_manager")
    
    if uploaded_file:
        logger.info(f"File uploaded: {uploaded_file.name}")
        db_path = save_uploaded_file(uploaded_file)
        logger.debug(f"Saved to temporary path: {db_path}")
        
        st.session_state.current_db_path = db_path
        tables = get_tables_from_db(db_path)
        logger.info(f"Available tables: {tables}")
        
        selected_table = st.selectbox("Select table to modify", tables)
        
        if selected_table:
            logger.info(f"Selected table: {selected_table}")
            st.session_state.current_table = selected_table
            df, types = read_table_from_db(db_path, selected_table)
            logger.debug(f"Initial DataFrame shape: {df.shape}")
            logger.debug(f"Initial columns: {df.columns.tolist()}")
            
            original_df = df.copy()
            st.session_state.current_df = df
            
            # Display original database statistics
            st.subheader("Original Database Statistics")
            stat_cols = st.columns(4)
            stat_cols[0].metric("Total Rows", df.shape[0])
            stat_cols[1].metric("Total Columns", df.shape[1])
            stat_cols[2].metric("Missing Values", df.isna().sum().sum())
            stat_cols[3].metric("Duplicate Rows", df.duplicated().sum())
            
            # Display column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Type': df.dtypes.astype(str),  # Convert dtype to string
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.subheader("Current Table Preview")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Column management
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Add New Column")
                new_col_name = st.text_input("New Column Name")
                col_type = st.selectbox("Column Type", ["int64", "float64", "string", "datetime64"])
                default_value = st.text_input("Default Value (optional)")
                
                if st.button("Add Column"):
                    if new_col_name and new_col_name not in df.columns:
                        try:
                            # Convert default value based on type
                            if col_type == "int64":
                                default_val = int(default_value) if default_value else 0
                            elif col_type == "float64":
                                default_val = float(default_value) if default_value else 0.0
                            elif col_type == "datetime64":
                                default_val = pd.to_datetime(default_value) if default_value else pd.NaT
                            else:
                                default_val = default_value if default_value else ""
                            
                            df[new_col_name] = default_val
                            st.session_state.current_df = df.copy()  # Store a copy
                            st.success(f"Column '{new_col_name}' added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding column: {str(e)}")
                    else:
                        st.error("Please provide a unique column name.")
            
            with col2:
                st.subheader("Remove Column")
                logger.info("Entering column removal section")
                
                # Debug current state
                logger.debug(f"Current DataFrame in session: {st.session_state.current_df.shape}")
                logger.debug(f"Available columns: {df.columns.tolist()}")
                
                # Store selected columns in session state
                if 'cols_to_remove' not in st.session_state:
                    st.session_state.cols_to_remove = []
                
                cols_to_remove = st.multiselect(
                    "Select columns to remove",
                    df.columns,
                    key='column_selector'
                )
                
                if cols_to_remove:
                    logger.info(f"Columns selected for removal: {cols_to_remove}")
                    st.session_state.cols_to_remove = cols_to_remove
                
                # Create a unique key for the button
                remove_cols_button = st.button(
                    "Remove Selected Columns",
                    key=f"remove_cols_button_{str(cols_to_remove)}"
                )
                
                logger.debug(f"Remove button clicked: {remove_cols_button}")
                logger.debug(f"Columns to remove: {st.session_state.cols_to_remove}")
                
                if remove_cols_button and st.session_state.cols_to_remove:
                    logger.info("Starting column removal process")
                    try:
                        # Get current DataFrame
                        current_df = st.session_state.current_df.copy()
                        logger.debug(f"Current DataFrame shape before removal: {current_df.shape}")
                        
                        # Remove columns
                        columns_to_remove = st.session_state.cols_to_remove
                        logger.info(f"Attempting to remove columns: {columns_to_remove}")
                        
                        new_df = current_df.drop(columns=columns_to_remove)
                        logger.info(f"Columns removed. New shape: {new_df.shape}")
                        
                        # Update session state
                        st.session_state.current_df = new_df
                        logger.info("Session state updated")
                        
                        # Show success message
                        success_msg = f"✅ Removed columns: {', '.join(columns_to_remove)}"
                        logger.info(success_msg)
                        st.success(success_msg)
                        
                        # Show immediate feedback
                        st.write("### Current Columns")
                        st.write(new_df.columns.tolist())
                        
                        # Clear the selection
                        st.session_state.cols_to_remove = []
                        
                        # Force refresh
                        logger.info("Forcing page refresh")
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Error during column removal: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error(f"❌ Error removing columns: {str(e)}")
                        st.write("Error details:", str(e))
                        import traceback
                        st.write("Traceback:", traceback.format_exc())
            
            # Compare current and original dataframe
            if st.session_state.current_df is not None and not st.session_state.current_df.equals(original_df):
                st.subheader("Changes Summary")
                changes_cols = st.columns(3)
                
                # Column changes
                current_cols = set(st.session_state.current_df.columns)
                original_cols = set(original_df.columns)
                added_cols = current_cols - original_cols
                removed_cols = original_cols - current_cols
                
                changes_cols[0].metric("Added Columns", len(added_cols))
                changes_cols[1].metric("Removed Columns", len(removed_cols))
                changes_cols[2].metric("Net Change", len(added_cols) - len(removed_cols))
                
                if added_cols:
                    st.write("Added columns:", ", ".join(added_cols))
                if removed_cols:
                    st.write("Removed columns:", ", ".join(removed_cols))
            
            # Save changes
            if st.button("Save Changes to Database"):
                logger.info("Save changes button clicked")
                if st.session_state.current_df is not None:
                    try:
                        current_df = st.session_state.current_df.copy()
                        logger.debug(f"Preparing to save DataFrame with shape: {current_df.shape}")
                        logger.debug(f"Columns to save: {current_df.columns.tolist()}")
                        
                        export_to_sqlite(
                            current_df, 
                            st.session_state.current_db_path, 
                            st.session_state.current_table
                        )
                        logger.info("Changes saved successfully to database")
                        
                        # Display final comparison
                        st.subheader("Final Database Comparison")
                        final_cols = st.columns(3)
                        with final_cols[0]:
                            st.metric("Original Shape", f"{original_df.shape[0]} × {original_df.shape[1]}")
                        with final_cols[1]:
                            st.metric("Final Shape", f"{current_df.shape[0]} × {current_df.shape[1]}")
                        with final_cols[2]:
                            col_diff = abs(current_df.shape[1] - original_df.shape[1])
                            st.metric("Columns Changed", col_diff, 
                                    delta=f"{'+' if current_df.shape[1] > original_df.shape[1] else '-'}{col_diff}")
                        
                        # Show column differences
                        added = set(current_df.columns) - set(original_df.columns)
                        removed = set(original_df.columns) - set(current_df.columns)
                        if added:
                            st.write("Added columns:", ", ".join(added))
                        if removed:
                            st.write("Removed columns:", ", ".join(removed))
                        
                        # Provide download option
                        with open(st.session_state.current_db_path, 'rb') as f:
                            st.download_button(
                                label="Download modified database",
                                data=f,
                                file_name="modified_database.db",
                                mime="application/x-sqlite3"
                            )
                    except Exception as e:
                        logger.error(f"Error saving changes: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error(f"Error saving changes: {str(e)}")
                else:
                    logger.warning("No DataFrame found in session state") 