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
            # Create a more detailed text representation
            parts = []
            for k, v in row.items():
                if pd.notna(v):
                    # Format numbers with consistent precision
                    if isinstance(v, (int, float)):
                        parts.append(f"{k}: {float(v):.2f}")
                    else:
                        # Convert string values to lowercase for better matching
                        parts.append(f"{k}: {str(v).lower()}")
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

    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      similarity_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
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
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)

            # Find matches with lower threshold
            matched_pairs = []
            used_indices_df2 = set()

            # Sort similarities in descending order for better matching
            similarity_pairs = []
            for idx1, similarities in enumerate(similarity_matrix):
                for idx2, score in enumerate(similarities):
                    if score >= similarity_threshold:
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

            # Rename df2's columns to match df1 where there are matches
            rename_map = {v: k for k, v in schema_matches.items()}
            df2_renamed = df2_renamed.rename(columns=rename_map)

            # Add match indicator columns
            df1_renamed['_merge_idx'] = range(len(df1_renamed))
            df2_renamed['_merge_idx'] = range(len(df2_renamed))

            # Perform full outer join
            merged_df = pd.concat([df1_renamed, df2_renamed], ignore_index=True)

            # Add match information
            merged_df['is_matched'] = False
            for idx1, idx2, score in matched_pairs:
                merged_df.loc[merged_df['_merge_idx'] == idx1, 'is_matched'] = True
                merged_df.loc[merged_df['_merge_idx'] == idx2, 'is_matched'] = True
                merged_df.loc[merged_df['_merge_idx'] == idx2, 'match_score'] = score

            # Drop temporary columns
            merged_df = merged_df.drop(columns=['_merge_idx'])

            # Compile statistics
            stats = {
                'total_matches': len(matched_pairs),
                'schema_matches': len(schema_matches),
                'unmatched_df1': len(df1) - len({p[0] for p in matched_pairs}),
                'unmatched_df2': len(df2) - len({p[1] for p in matched_pairs}),
                'matched_pairs': matched_pairs
            }

            logger.info(f"Merge complete. Found {len(matched_pairs)} matching records")
            return merged_df, stats
            
        except Exception as e:
            logger.error(f"Error in enhanced merge process: {str(e)}")
            raise