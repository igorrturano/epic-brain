from config import RAW_DATA_DIR, VECTOR_STORE_DIR, setup_logging, create_directories
from ingestion_pipeline import IngestionPipeline
from pathlib import Path
import os
import sys
import pandas as pd

logger = setup_logging()

def process_documents():
    try:
        # Create necessary directories
        create_directories()
        
        # Check if raw directory exists
        if not RAW_DATA_DIR.exists():
            logger.error(f"Raw data directory does not exist: {RAW_DATA_DIR}")
            return
            
        # Check if cleaned logs file exists
        cleaned_logs_path = RAW_DATA_DIR / "cleaned_logs.csv"
        if not cleaned_logs_path.exists():
            logger.error(f"Cleaned logs file does not exist: {cleaned_logs_path}")
            return
            
        # Initialize pipeline
        pipeline = IngestionPipeline(
            raw_directory=RAW_DATA_DIR,
            vector_store_path=VECTOR_STORE_DIR
        )
        
        # Read the cleaned logs
        try:
            df = pd.read_csv(cleaned_logs_path)
            logger.info(f"Found {len(df)} entries in cleaned logs")
        except Exception as e:
            logger.error(f"Error reading cleaned logs: {e}")
            return
        
        # Process the cleaned logs
        logger.info("\nProcessing cleaned logs...")
        pipeline.process_dataframe(df)
        
        # Verify vector store
        vector_store = pipeline.get_vector_store()
        if vector_store:
            logger.info("Vector store created successfully")
            # Get collection info
            collection = vector_store._collection
            logger.info(f"Number of collections in vector store: {collection.count()}")
        else:
            logger.error("Failed to create vector store")
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise

if __name__ == "__main__":
    # Set stdout encoding to utf-8
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    process_documents() 