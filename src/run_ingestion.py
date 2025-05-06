import argparse
from pathlib import Path
from ingestion_pipeline import IngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Process raw files and create vector store")
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Directory containing raw files"
    )
    parser.add_argument(
        "--vector_store_dir",
        type=str,
        required=True,
        help="Directory to store the vector database"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of text chunks for splitting"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(args.vector_store_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the pipeline
    pipeline = IngestionPipeline(
        raw_directory=args.raw_dir,
        vector_store_path=args.vector_store_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline.process_directory()

if __name__ == "__main__":
    main() 