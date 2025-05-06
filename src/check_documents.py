from pathlib import Path
from config import RAW_DATA_DIR, VECTOR_STORE_DIR, setup_logging, create_directories
from rag.pipeline import RAGPipeline
from rag.retriever import ChromaRetriever
from ingestion_pipeline import IngestionPipeline

logger = setup_logging()

def check_document_processing():
    try:
        # Create necessary directories
        create_directories()
        
        # Initialize pipelines
        ingestion_pipeline = IngestionPipeline()
        retriever = ChromaRetriever(vector_store_path=VECTOR_STORE_DIR)
        rag_pipeline = RAGPipeline(retriever=retriever)
        
        # Check raw directory
        raw_files = list(RAW_DATA_DIR.glob("*.md"))
        logger.info(f"Found {len(raw_files)} raw files in {RAW_DATA_DIR}")
        
        # Check vector store
        if VECTOR_STORE_DIR.exists():
            logger.info("Vector store exists, testing query...")
            test_query = "What is the main topic of the documents?"
            results = rag_pipeline.query(test_query)
            logger.info(f"Test query results: {results}")
        else:
            logger.info("Vector store does not exist yet")
            
    except Exception as e:
        logger.error(f"Error checking documents: {e}")
        raise

if __name__ == "__main__":
    check_document_processing() 