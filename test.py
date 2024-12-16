import streamlit as st
import pandas as pd
import numpy as np
from merger import DatasetMerger
from preprocessor import DataPreprocessor
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(uploaded_file):
    """Load data from uploaded file."""
    try:
        # Try to read as CSV first
        df = pd.read_csv(uploaded_file)
        return df
    except:
        try:
            # Try to read as Excel if CSV fails
            df = pd.read_excel(uploaded_file)
            return df
        except:
            st.error(f"Failed to load file {uploaded_file.name}. Please ensure it's a valid CSV or Excel file.")
            return None

def show_dataframe_analysis(df, title):
    """Show detailed analysis of a dataframe."""
    st.write(f"#### {title}")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    
    # Data types and missing values
    dtype_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.write("Data Types and Missing Values:")
    st.dataframe(dtype_df)
    
    # Preview data
    st.write("Data Preview:")
    st.dataframe(df.head())

def main():
    st.title("Dataset Merger Testing Tool")
    st.write("""
    Upload two datasets (CSV or Excel files) to test the preprocessing and merger functionality.
    The tool will show detailed information about schema matching, data cleaning, and merging steps.
    """)

    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Dataset")
        file1 = st.file_uploader("Upload first file", type=['csv', 'xlsx'])
        
    with col2:
        st.subheader("Second Dataset")
        file2 = st.file_uploader("Upload second file", type=['csv', 'xlsx'])

    if file1 is not None and file2 is not None:
        # Load datasets
        df1 = load_data(file1)
        df2 = load_data(file2)
        
        if df1 is not None and df2 is not None:
            st.write("## Original Dataset Information")
            
            # Show original dataset info
            col1, col2 = st.columns(2)
            with col1:
                show_dataframe_analysis(df1, "First Dataset")
            with col2:
                show_dataframe_analysis(df2, "Second Dataset")
            
            # Add preprocessing and merge button
            if st.button("Preprocess and Merge Datasets"):
                try:
                    st.write("## Preprocessing Steps")
                    
                    # Initialize preprocessor and merger
                    preprocessor = DataPreprocessor()
                    merger = DatasetMerger()
                    
                    with st.spinner("Preprocessing datasets..."):
                        # Preprocessing step
                        st.write("### 1. Data Cleaning and Preprocessing")
                        df1_processed, df2_processed, schema_matches = preprocessor.preprocess_datasets(df1, df2)
                        
                        # Show preprocessing results
                        st.write("#### Preprocessing Results:")
                        col1, col2 = st.columns(2)
                        with col1:
                            show_dataframe_analysis(df1_processed, "Preprocessed First Dataset")
                        with col2:
                            show_dataframe_analysis(df2_processed, "Preprocessed Second Dataset")
                        
                        # Show schema matching results
                        st.write("### 2. Schema Matching Results")
                        schema_df = pd.DataFrame(list(schema_matches.items()), 
                                              columns=['Dataset 1 Column', 'Dataset 2 Column'])
                        st.write("Matched Columns:")
                        st.dataframe(schema_df)
                        
                        # Show unmatched columns
                        unmatched_df1 = set(df1.columns) - set(schema_matches.keys())
                        unmatched_df2 = set(df2.columns) - set(schema_matches.values())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Unmatched Columns in Dataset 1:")
                            st.write(list(unmatched_df1))
                        with col2:
                            st.write("Unmatched Columns in Dataset 2:")
                            st.write(list(unmatched_df2))
                    
                    with st.spinner("Merging datasets..."):
                        # Merging step
                        st.write("### 3. Merging Datasets")
                        merged_df, stats = merger.merge_datasets(df1, df2)
                        
                        # Display merge statistics
                        st.write("#### Merge Statistics:")
                        st.write("- Total matches found:", stats['total_matches'])
                        st.write("- Records from Dataset 1:", len(df1))
                        st.write("- Records from Dataset 2:", len(df2))
                        st.write("- Unmatched in Dataset 1:", stats['unmatched_df1'])
                        st.write("- Unmatched in Dataset 2:", stats['unmatched_df2'])
                        
                        # Show matched pairs details
                        if stats['matched_pairs']:
                            st.write("#### Matched Record Details:")
                            matched_pairs_df = pd.DataFrame(stats['matched_pairs'], 
                                                         columns=['Index in DF1', 'Index in DF2', 'Match Score'])
                            st.dataframe(matched_pairs_df)
                        
                        # Show final merged dataset
                        st.write("### 4. Final Merged Dataset")
                        show_dataframe_analysis(merged_df, "Merged Dataset")
                        
                        # Add download button for merged dataset
                        csv = merged_df.to_csv(index=False)
                        st.download_button(
                            label="Download Merged Dataset",
                            data=csv,
                            file_name="merged_dataset.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    logger.exception("Error details:")

if __name__ == "__main__":
    main() 