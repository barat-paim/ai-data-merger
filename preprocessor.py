import jellyfish
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, jellyfish_model_path=None):
        """Initialize the data preprocessor.
        
        Args:
            jellyfish_model_path (str, optional): Path to jellyfish model. Defaults to None.
        """
        self.jellyfish_model_path = jellyfish_model_path
        # Initialize BERT model for semantic matching
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initialized DataPreprocessor with BERT and Jellyfish models")

    def _enhance_column_name(self, col: str) -> str:
        """Add context to column names by expanding common abbreviations and adding semantic context."""
        col = col.lower().replace('_', ' ')
        if 'id' in col:
            col += ' identifier primary key'
        if any(age in col for age in ['age', 'years']):
            col += ' age in years numeric'
        if any(name in col for name in ['name', 'firstname', 'lastname']):
            col += ' person name text'
        return col

    def _match_columns(self, df1_columns: List[str], df2_columns: List[str]) -> Dict[str, str]:
        """Match columns using both BERT embeddings and Jellyfish string similarity."""
        matches = {}
        
        # Prepare enhanced column descriptions
        df1_enhanced = [self._enhance_column_name(col) for col in df1_columns]
        df2_enhanced = [self._enhance_column_name(col) for col in df2_columns]
        
        # Get BERT embeddings for enhanced column descriptions
        embeddings1 = self.bert_model.encode(df1_enhanced)
        embeddings2 = self.bert_model.encode(df2_enhanced)
        
        # Calculate similarity matrix using both BERT and Jellyfish
        similarity_matrix = np.zeros((len(df1_columns), len(df2_columns)))
        
        for i, col1 in enumerate(df1_columns):
            for j, col2 in enumerate(df2_columns):
                # BERT similarity (cosine similarity of embeddings)
                bert_sim = cosine_similarity([embeddings1[i]], [embeddings2[j]])[0][0]
                
                # Jellyfish similarity
                jelly_sim = jellyfish.jaro_winkler_similarity(col1.lower(), col2.lower())
                
                # Combine similarities (weighted average)
                similarity_matrix[i, j] = 0.7 * bert_sim + 0.3 * jelly_sim

        # Find best matches using similarity matrix
        threshold = 0.6  # Minimum similarity threshold
        used_cols2 = set()
        
        for i, col1 in enumerate(df1_columns):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_match_score = similarity_matrix[i][best_match_idx]
            
            if best_match_score >= threshold and best_match_idx not in used_cols2:
                matches[col1] = df2_columns[best_match_idx]
                used_cols2.add(best_match_idx)
                logger.debug(f"Matched columns: {col1} -> {df2_columns[best_match_idx]} (score: {best_match_score:.2f})")
        
        return matches

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
            
            # Match schemas using improved matching
            schema_matches = self._match_columns(list(df1_imputed.columns), list(df2_imputed.columns))
            logger.info(f"Found {len(schema_matches)} column matches")
            
            return df1_imputed, df2_imputed, schema_matches
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
