import jellyfish
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with jellyfish string matching."""
        logger.info("Initialized DataPreprocessor with jellyfish string matching")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing duplicates and standardizing formats."""
        logger.info("Starting data cleaning process")
        try:
            # Remove exact duplicates
            df = df.drop_duplicates()
            
            # Standardize string columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip().str.lower()
            
            logger.info(f"Cleaned data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple imputation of missing values."""
        logger.info("Starting missing value imputation")
        try:
            # For numeric columns, use median
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)

            # For string columns, use mode
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '', inplace=True)
            
            logger.info("Completed missing value imputation")
            return df
        except Exception as e:
            logger.error(f"Error in value imputation: {str(e)}")
            raise

    def match_schemas(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, str]:
        """Match columns between two dataframes using jellyfish string matching."""
        logger.info("Starting schema matching")
        try:
            column_matches = {}
            for col1 in df1.columns:
                best_match = None
                best_score = 0
                
                # Get sample values from first dataframe
                sample_values1 = df1[col1].dropna().head(5).astype(str).tolist()
                
                for col2 in df2.columns:
                    # Calculate name similarity using multiple metrics
                    name_jaro = jellyfish.jaro_winkler_similarity(col1, col2)
                    name_levenshtein = 1 - (jellyfish.levenshtein_distance(col1, col2) / 
                                          max(len(col1), len(col2)))
                    
                    # Get sample values from second dataframe
                    sample_values2 = df2[col2].dropna().head(5).astype(str).tolist()
                    
                    # Calculate value similarity
                    value_similarities = []
                    for val1 in sample_values1:
                        for val2 in sample_values2:
                            value_similarities.append(
                                jellyfish.jaro_winkler_similarity(str(val1), str(val2))
                            )
                    value_similarity = np.mean(value_similarities) if value_similarities else 0
                    
                    # Combine similarities (weighted average)
                    match_score = (0.4 * name_jaro + 
                                 0.3 * name_levenshtein + 
                                 0.3 * value_similarity)
                    
                    if match_score > best_score:
                        best_score = match_score
                        best_match = col2
                
                if best_score > 0.7:  # Configurable threshold
                    column_matches[col1] = best_match
                    logger.debug(f"Matched columns: {col1} -> {best_match} (score: {best_score:.2f})")
            
            logger.info(f"Found {len(column_matches)} column matches")
            return column_matches
        except Exception as e:
            logger.error(f"Error in schema matching: {str(e)}")
            raise

    def preprocess_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
        """Complete preprocessing pipeline for two datasets."""
        logger.info("Starting complete preprocessing pipeline")
        try:
            # Clean both datasets
            df1_cleaned = self.clean_data(df1.copy())
            df2_cleaned = self.clean_data(df2.copy())
            
            # Impute missing values
            df1_imputed = self.impute_missing_values(df1_cleaned)
            df2_imputed = self.impute_missing_values(df2_cleaned)
            
            # Match schemas
            schema_matches = self.match_schemas(df1_imputed, df2_imputed)
            
            logger.info("Completed preprocessing pipeline")
            return df1_imputed, df2_imputed, schema_matches
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
