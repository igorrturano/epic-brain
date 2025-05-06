import os
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import logging

from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import torch

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(
        self,
        raw_directory: str,
        vector_store_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            raw_directory: Path to directory containing raw files
            vector_store_path: Path to store the vector database
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model_name: Name of the embedding model to use
        """
        self.raw_directory = Path(raw_directory)
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Prioritize splitting on paragraph boundaries
            keep_separator=True  # Keep the separators in the chunks
        )
        
        # Initialize embeddings with GPU if available
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": self.device}
        )
        
        # Initialize vector store
        self.vector_store = None
        
    def process_raw(self, raw_path: Path) -> List[Document]:
        """
        Process a single raw file and convert it to LangChain documents.
        
        Args:
            raw_path: Path to the raw file
            
        Returns:
            List of LangChain documents
        """
        try:
            # Read the log file with proper encoding handling
            try:
                with open(raw_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 fails
                with open(raw_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Extract metadata from path
            # Path format: data/raw/players/date/player_name.log
            parts = raw_path.relative_to(self.raw_directory).parts
            date = parts[1]  # players/date
            player_name = raw_path.stem  # Get player name from filename without extension
            
            # Process log entries
            documents = []
            for line in content.splitlines():
                if not line.strip():
                    continue
                    
                try:
                    # Parse log entry
                    # Format: date,account,race,location,text,characters_proximity
                    timestamp, account, race, location, text, proximity = line.strip().split(',', 5)
                    
                    # Create a context-rich text for better semantic search
                    context_text = f"Em {timestamp}, {account} ({race}) estava em {location}"
                    if proximity and proximity != "None":
                        context_text += f" próximo de {proximity}"
                    context_text += f". {account} disse: {text}"
                    
                    # Create metadata with all relevant information
                    # Convert None values to empty strings for Chroma compatibility
                    metadata = {
                        "source": str(raw_path),
                        "filename": raw_path.name,
                        "player": player_name,
                        "date": date,
                        "timestamp": timestamp,
                        "account": account,
                        "race": race,
                        "location": location,
                        "proximity": proximity if proximity != "None" else ""
                    }
                    
                    # Create document with context-rich text and metadata
                    doc = Document(
                        page_content=context_text,
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                except ValueError as e:
                    logger.warning(f"Could not parse line in {raw_path}: {line.strip()}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing {raw_path}: {str(e)}")
            return []
    
    def process_directory(self) -> None:
        """
        Process all raw files in the directory and store them in the vector database.
        """
        # Get all raw files from subdirectories
        raw_files = []
        for date_dir in self.raw_directory.glob("players/*"):
            if not date_dir.is_dir():
                continue
            raw_files.extend(list(date_dir.glob("*.log")))
        
        if not raw_files:
            logger.warning(f"No raw files found in {self.raw_directory}")
            return
        
        # Process each raw file
        all_documents = []
        for raw_file in tqdm(raw_files, desc="Processing RAWs"):
            documents = self.process_raw(raw_file)
            all_documents.extend(documents)
        
        if not all_documents:
            logger.warning("No documents were processed successfully")
            return
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=str(self.vector_store_path)
        )
        
        logger.info(f"Processed {len(all_documents)} chunks from {len(raw_files)} raw files")
        
    def get_vector_store(self) -> Optional[Chroma]:
        """
        Get the vector store instance.
        
        Returns:
            Chroma vector store instance or None if not initialized
        """
        if self.vector_store is None:
            try:
                # Try to load existing vector store
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_store_path),
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {self.vector_store_path}")
            except Exception as e:
                logger.error(f"Could not load vector store: {e}")
                return None
        return self.vector_store 