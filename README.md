# ML-Powered Dataset Merger

A powerful Streamlit application that intelligently merges SQLite databases using machine learning. The app uses BERT-based semantic similarity matching to identify and merge related records across different databases, even when they have different schemas.

## Features

- **Intelligent Merging**: Uses the `all-MiniLM-L6-v2` BERT model to compute semantic similarity between records
- **Flexible Schema Handling**: Can merge databases with different schemas, preserving all columns
- **Interactive UI**: Clear visualization of data and merge statistics
- **Comprehensive Logging**: Detailed logging of all operations for debugging
- **Export Options**: 
  - Save merged databases locally with timestamps
  - Download merged databases directly
  - View merge statistics and previews

## Requirements

```bash
streamlit
pandas
sentence-transformers
torch
scikit-learn
sqlite3
```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Upload two SQLite databases
3. Select tables to merge
4. Adjust similarity threshold if needed
5. Click "Merge Datasets" to start the process
6. Export the merged database in your preferred format

## Key Achievements

1. **Semantic Matching**: Successfully implemented ML-based record matching using BERT embeddings
2. **Flexible Schema Support**: 
   - Handles databases with different column structures
   - Preserves unique columns from both sources
   - Intelligently matches based on common columns
3. **Robust Error Handling**:
   - Validates database compatibility
   - Provides clear error messages
   - Includes comprehensive logging
4. **User Experience**:
   - Interactive data previews
   - Clear merge statistics
   - Progress indicators
   - Multiple export options

## Output

Merged databases are saved in the `merged_databases` directory with timestamps for easy tracking. Each export includes:
- The merged SQLite database
- Detailed merge statistics
- Logging information in `app.log`

## Notes

- The similarity threshold can be adjusted (0.0 to 1.0) to control matching strictness
- Higher thresholds result in more precise but fewer matches
- Lower thresholds increase matches but may include less certain pairs
```

This README provides a comprehensive overview of:
1. What the application does
2. Its key features and achievements
3. How to use it
4. Technical requirements
5. Important implementation details

Would you like me to add or modify any specific section of the README?