import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import torch
from tqdm import tqdm
import warnings
import logging
from preprocessor import DataPreprocessor
import jellyfish

# Configure logger
logger = logging.getLogger(__name__)

# Suppress specific PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
logging.getLogger('transformers').setLevel(logging.ERROR)

class DatasetMerger:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', jellyfish_model_path: Optional[str] = None):
        """Initialize the dataset merger with both BERT and Jellyfish models."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bert_model = SentenceTransformer(model_name)
        
        self.preprocessor = DataPreprocessor(jellyfish_model_path)
        logger.info(f"Initialized DatasetMerger with model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_grad_enabled(False)
        self.bert_model.to(self.device)

    def _prepare_text_representation(self, df: pd.DataFrame) -> List[str]:
        """Convert DataFrame rows to text representation."""
        text_representations = []
        for _, row in df.iterrows():
            # Create a more detailed text representation
            parts = []
            for k, v in row.items():
                if pd.notna(v):
                    # Format numbers with consistent precision
                    if isinstance(v, (int, float)):
                        parts.append(f"{k}: {float(v):.2f}")
                    else:
                        # Convert string values to lowercase and normalize
                        v_str = str(v).lower().strip()
                        # Add more weight to ID fields
                        if 'id' in k.lower():
                            parts.extend([f"{k}: {v_str}"] * 3)  # Repeat ID fields for more weight
                        else:
                            parts.append(f"{k}: {v_str}")
            row_text = ' | '.join(parts)
            text_representations.append(row_text)
        return text_representations

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for text representations."""
        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.bert_model.encode(batch)
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _calculate_field_similarity(self, row1, row2, schema_matches):
        """Calculate field-by-field similarity between two rows."""
        similarities = []
        weights = []
        
        for col1, col2 in schema_matches.items():
            val1 = row1[col1]
            val2 = row2[col2]
            
            if pd.isna(val1) or pd.isna(val2):
                continue
                
            # Convert values to strings for comparison
            str_val1 = str(val1).lower().strip()
            str_val2 = str(val2).lower().strip()
            
            # Calculate field similarity based on type
            if 'id' in col1.lower():
                # Exact match for IDs
                sim = 1.0 if str_val1 == str_val2 else 0.0
                weight = 5.0  # High weight for ID matches
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                sim = 1.0 - (abs(val1 - val2) / max_val if max_val > 0 else 0)
                weight = 2.0
            else:
                # String similarity using Jaro-Winkler
                sim = jellyfish.jaro_winkler_similarity(str_val1, str_val2)
                weight = 1.0
            
            similarities.append(sim)
            weights.append(weight)
        
        if not similarities:
            return 0.0
            
        return np.average(similarities, weights=weights)

    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      similarity_threshold: float = 0.85) -> Tuple[pd.DataFrame, Dict]:
        """Merge two datasets using both Jellyfish preprocessing and BERT-based matching."""
        logger.info("Starting enhanced dataset merge process")
        
        try:
            # Preprocess datasets using Jellyfish
            df1_processed, df2_processed, schema_matches = self.preprocessor.preprocess_datasets(df1, df2)
            logger.info(f"Preprocessing complete. Schema matches found: {schema_matches}")
            
            # Use schema matches to align columns for comparison
            common_columns = list(schema_matches.keys())
            
            # Calculate unique columns for each dataset
            unique_cols_df1 = list(set(df1.columns) - set(common_columns))
            unique_cols_df2 = list(set(df2.columns) - set(schema_matches.values()))
            
            # Use only common columns for matching
            df1_common = df1_processed[common_columns].copy()
            df2_common = df2_processed[[schema_matches[col] for col in common_columns]].copy()
            
            # Prepare text representations using aligned columns
            texts1 = self._prepare_text_representation(df1_common)
            texts2 = self._prepare_text_representation(df2_common)
            
            # Compute embeddings
            print("Computing embeddings for first dataset...")
            embeddings1 = self._compute_embeddings(texts1)
            print("Computing embeddings for second dataset...")
            embeddings2 = self._compute_embeddings(texts2)

            # Compute similarity matrix
            print("Computing similarity matrix...")
            bert_similarity_matrix = cosine_similarity(embeddings1, embeddings2)

            # Calculate field-by-field similarity matrix
            field_similarity_matrix = np.zeros((len(df1), len(df2)))
            for i in range(len(df1)):
                for j in range(len(df2)):
                    field_similarity_matrix[i, j] = self._calculate_field_similarity(
                        df1_processed.iloc[i], df2_processed.iloc[j], schema_matches
                    )

            # Combine similarities with weights
            similarity_matrix = 0.3 * bert_similarity_matrix + 0.7 * field_similarity_matrix

            # Find matches with higher threshold
            matched_pairs = []
            used_indices_df2 = set()

            # Sort similarities in descending order for better matching
            similarity_pairs = []
            for idx1, similarities in enumerate(similarity_matrix):
                for idx2, score in enumerate(similarities):
                    if score >= similarity_threshold:
                        # Additional check for ID match if available
                        id_match = False
                        for col1, col2 in schema_matches.items():
                            if 'id' in col1.lower():
                                val1 = str(df1_processed.iloc[idx1][col1]).lower().strip()
                                val2 = str(df2_processed.iloc[idx2][col2]).lower().strip()
                                if val1 == val2:
                                    id_match = True
                                    break
                        
                        if id_match:
                            score = 1.0  # Boost score for ID matches
                        
                        similarity_pairs.append((score, idx1, idx2))
            
            # Sort by similarity score in descending order
            similarity_pairs.sort(reverse=True)
            
            # Assign matches greedily from highest similarity to lowest
            for score, idx1, idx2 in similarity_pairs:
                if idx1 not in {p[0] for p in matched_pairs} and idx2 not in used_indices_df2:
                    matched_pairs.append((idx1, idx2, score))
                    used_indices_df2.add(idx2)

            # Create a new DataFrame for each input with standardized column names
            df1_renamed = df1.copy()
            df2_renamed = df2.copy()

            # Keep track of original columns
            df1_cols = set(df1.columns)
            df2_cols = set(df2.columns)

            # Create copies of columns that will be renamed, to preserve original names
            for v, k in schema_matches.items():
                if k in df2_renamed.columns:
                    df2_renamed[k] = df2_renamed[k].copy()  # Preserve original column
                    df2_renamed[v] = df2_renamed[k].copy()  # Add mapped column

            # Add columns that exist only in df1 to df2 (with NaN values)
            for col in df1_cols - set(df2_renamed.columns):
                df2_renamed[col] = np.nan

            # Add columns that exist only in df2 to df1 (with NaN values)
            for col in df2_cols - set(df1_renamed.columns):
                df1_renamed[col] = np.nan

            # Add match indicator columns
            df1_renamed['_merge_idx'] = range(len(df1_renamed))
            df2_renamed['_merge_idx'] = range(len(df2_renamed))

            # Perform full outer join
            merged_df = pd.concat([df1_renamed, df2_renamed], ignore_index=True)

            # Add match information
            merged_df['is_matched'] = False
            merged_df['match_score'] = np.nan
            for idx1, idx2, score in matched_pairs:
                merged_df.loc[merged_df['_merge_idx'] == idx1, 'is_matched'] = True
                merged_df.loc[merged_df['_merge_idx'] == idx2, 'is_matched'] = True
                merged_df.loc[merged_df['_merge_idx'] == idx2, 'match_score'] = score

            # Drop temporary columns
            merged_df = merged_df.drop(columns=['_merge_idx'])

            # Final imputation step to handle any remaining NaN values
            logger.info("Performing final imputation on merged dataset")

            # Debug: Print columns with NaN values before imputation
            nan_cols = merged_df.columns[merged_df.isna().any()].tolist()
            logger.info(f"Columns with NaN values before imputation: {nan_cols}")

            # Handle missing values for each column type appropriately
            for col in merged_df.columns:
                if col in ['is_matched', 'match_score']:  # Skip metadata columns
                    continue

                if merged_df[col].isna().any():
                    logger.info(f"Imputing column: {col}")
                    if merged_df[col].dtype.kind in 'bifc':  # Numeric columns
                        # For numeric columns, use median for imputation
                        median_value = merged_df[col].median()
                        if pd.isna(median_value):  # If median is also NaN
                            merged_df[col] = merged_df[col].fillna(0)
                        else:
                            merged_df[col] = merged_df[col].fillna(median_value)
                    else:  # Non-numeric columns (strings, objects)
                        # For categorical/string columns, use mode or empty string
                        mode_values = merged_df[col].mode()
                        if len(mode_values) > 0 and not pd.isna(mode_values[0]):
                            merged_df[col] = merged_df[col].fillna(mode_values[0])
                        else:
                            # If no mode exists or mode is NaN, use empty string
                            merged_df[col] = merged_df[col].fillna('')

            # Debug: Print columns with NaN values after imputation
            nan_cols_after = [col for col in merged_df.columns if col != 'match_score' and merged_df[col].isna().any()]
            if nan_cols_after:
                logger.error(f"Columns still containing NaN values after imputation: {nan_cols_after}")
                for col in nan_cols_after:
                    nan_count = merged_df[col].isna().sum()
                    logger.error(f"Column {col} has {nan_count} NaN values")

            # Verify no NaN values remain (except in match_score)
            non_score_cols = [col for col in merged_df.columns if col != 'match_score']
            assert not merged_df[non_score_cols].isna().any().any(), "Found remaining NaN values after imputation"

            # Create stats dictionary
            stats = {
                'total_matches': len(matched_pairs),
                'schema_matches': schema_matches,
                'matched_pairs': matched_pairs,
                'unmatched_df1': len(df1) - len({p[0] for p in matched_pairs}),
                'unmatched_df2': len(df2) - len({p[1] for p in matched_pairs})
            }

            return merged_df, stats

        except Exception as e:
            logger.error(f"Error in enhanced merge process: {str(e)}")
            raise