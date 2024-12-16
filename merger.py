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
            # Convert row to string, excluding index
            row_text = ' '.join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
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

    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      similarity_threshold: float = 0.8) -> Tuple[pd.DataFrame, Dict]:
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
            
            df1_common = df1_processed[common_columns]
            df2_common = df2_processed[[schema_matches[col] for col in common_columns]]
            
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
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)

            # Find matches
            matched_pairs = []
            used_indices_df2 = set()

            for idx1, similarities in enumerate(similarity_matrix):
                best_match_idx = np.argmax(similarities)
                best_match_score = similarities[best_match_idx]
                
                if best_match_score >= similarity_threshold and best_match_idx not in used_indices_df2:
                    matched_pairs.append((idx1, best_match_idx, best_match_score))
                    used_indices_df2.add(best_match_idx)

            # Create merged dataset
            merged_rows = []
            unmatched_df1_indices = set(range(len(df1))) - set(p[0] for p in matched_pairs)
            unmatched_df2_indices = set(range(len(df2))) - set(p[1] for p in matched_pairs)

            # Initialize merged DataFrame with all possible columns
            all_columns = list(set(df1.columns) | set(df2.columns))
            merged_df = pd.DataFrame(columns=all_columns)

            # Add matched rows
            for idx1, idx2, score in matched_pairs:
                merged_row = {}
                # Add data from df1
                for col in df1.columns:
                    merged_row[col] = df1.iloc[idx1][col]
                # Add data from df2 for non-overlapping columns
                for col2 in df2.columns:
                    if col2 not in schema_matches.values():
                        merged_row[col2] = df2.iloc[idx2][col2]
                merged_rows.append(merged_row)

            # Add unmatched rows from df1
            for idx in unmatched_df1_indices:
                merged_row = {col: df1.iloc[idx][col] if col in df1.columns else None 
                            for col in all_columns}
                merged_rows.append(merged_row)

            # Add unmatched rows from df2
            for idx in unmatched_df2_indices:
                merged_row = {col: df2.iloc[idx][col] if col in df2.columns else None 
                            for col in all_columns}
                merged_rows.append(merged_row)

            # Create final merged DataFrame
            merged_df = pd.DataFrame(merged_rows)

            # Compile statistics
            stats = {
                'total_matches': len(matched_pairs),
                'schema_matches': len(schema_matches),
                'unmatched_df1': len(unmatched_df1_indices),
                'unmatched_df2': len(unmatched_df2_indices),
                'matched_pairs': matched_pairs
            }

            logger.info(f"Merge complete. Found {len(matched_pairs)} matching records")
            return merged_df, stats
            
        except Exception as e:
            logger.error(f"Error in enhanced merge process: {str(e)}")
            raise