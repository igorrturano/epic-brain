from config import RAW_DATA_DIR, VECTOR_STORE_DIR, setup_logging, create_directories
from ingestion_pipeline import IngestionPipeline
from pathlib import Path
import os
import sys

logger = setup_logging()

def process_documents():
    try:
        # Create necessary directories
        create_directories()
        
        # Check if raw directory exists
        if not RAW_DATA_DIR.exists():
            logger.error(f"Raw data directory does not exist: {RAW_DATA_DIR}")
            return
            
        # Check if players directory exists
        players_dir = RAW_DATA_DIR / "players"
        if not players_dir.exists():
            logger.error(f"Players directory does not exist: {players_dir}")
            return
            
        # Initialize pipeline
        pipeline = IngestionPipeline(
            raw_directory=RAW_DATA_DIR,
            vector_store_path=VECTOR_STORE_DIR
        )
        
        # Check raw directory and its subdirectories
        raw_files = []
        for date_dir in players_dir.glob("*"):
            if not date_dir.is_dir():
                continue
            logger.info(f"Checking date directory: {date_dir.name}")
            log_files = list(date_dir.glob("*.log"))
            raw_files.extend(log_files)
            for log_file in log_files:
                try:
                    logger.info(f"  Found log file: {log_file.name}")
                except UnicodeEncodeError:
                    # Fallback to ASCII representation if Unicode logging fails
                    logger.info(f"  Found log file: {log_file.name.encode('ascii', 'replace').decode()}")
        
        logger.info(f"\nFound {len(raw_files)} raw files in {RAW_DATA_DIR}:")
        for raw in raw_files:
            try:
                logger.info(f"- {raw.relative_to(RAW_DATA_DIR)}")
            except UnicodeEncodeError:
                # Fallback to ASCII representation if Unicode logging fails
                logger.info(f"- {str(raw.relative_to(RAW_DATA_DIR)).encode('ascii', 'replace').decode()}")
        
        if not raw_files:
            logger.warning("No log files found. Please check if files exist in the correct structure:")
            logger.warning("data/raw/players/[date]/[player_name].log")
            return
        
        # Process all raws
        logger.info("\nProcessing RAWs...")
        pipeline.process_directory()
        
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