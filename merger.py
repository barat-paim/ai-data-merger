import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import torch
from tqdm import tqdm
import warnings
import logging

# Suppress specific PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
logging.getLogger('transformers').setLevel(logging.ERROR)

class DatasetMerger:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the dataset merger with a specific sentence transformer model."""
        # Suppress torch warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SentenceTransformer(model_name)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Move model to device without logging
        torch.set_grad_enabled(False)
        self.model.to(self.device)

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
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      similarity_threshold: float = 0.8) -> Tuple[pd.DataFrame, Dict]:
        """
        Merge two datasets based on semantic similarity.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            Tuple of merged DataFrame and merge statistics
        """
        # Find common columns
        common_columns = list(set(df1.columns).intersection(set(df2.columns)))
        if not common_columns:
            raise ValueError("No common columns found between datasets")

        # Get unique columns from each dataset
        unique_cols_df1 = list(set(df1.columns) - set(df2.columns))
        unique_cols_df2 = list(set(df2.columns) - set(df1.columns))

        # Use only common columns for similarity comparison
        df1_common = df1[common_columns]
        df2_common = df2[common_columns]

        # Prepare text representations using only common columns
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
        for idx1, idx2, _ in matched_pairs:
            merged_row = {}
            # Add common columns from df1
            for col in common_columns:
                merged_row[col] = df1.iloc[idx1][col]
            # Add unique columns from both dataframes
            for col in unique_cols_df1:
                merged_row[col] = df1.iloc[idx1][col]
            for col in unique_cols_df2:
                merged_row[col] = df2.iloc[idx2][col]
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

        merged_df = pd.DataFrame(merged_rows, columns=all_columns)

        # Compile statistics
        stats = {
            'total_matches': len(matched_pairs),
            'unmatched_df1': len(unmatched_df1_indices),
            'unmatched_df2': len(unmatched_df2_indices),
            'total_rows': len(merged_df),
            'common_columns': common_columns,
            'unique_cols_df1': unique_cols_df1,
            'unique_cols_df2': unique_cols_df2,
            'matched_pairs': matched_pairs
        }

        return merged_df, stats