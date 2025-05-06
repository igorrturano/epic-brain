from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retriever.
        
        Args:
            config: Retriever configuration dictionary
        """
        self.config = config
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the retriever. Must be implemented by subclasses.
        """
        pass
        
    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents for a query. Must be implemented by subclasses.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        pass
        
    def is_initialized(self) -> bool:
        """
        Check if the retriever is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._is_initialized
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get the retriever configuration.
        
        Returns:
            Retriever configuration dictionary
        """
        return self.config.copy()

class Reranker:
    """
    Reranker implementation using cross-encoders for improved retrieval accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reranker.
        
        Args:
            config: Reranker configuration dictionary
        """
        self.config = config
        self.model_name = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_initialized = False
        
    def initialize(self) -> None:
        """
        Initialize the reranker model.
        """
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            self._is_initialized = True
            logger.info(f"Initialized reranker with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise
            
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents
        """
        if not self._is_initialized:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")
            
        try:
            # Prepare document pairs for scoring
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Sort documents by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k documents
            return [doc for doc, _ in scored_docs[:top_k]]
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise
            
    def is_initialized(self) -> bool:
        """
        Check if the reranker is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._is_initialized

class ChromaRetriever(BaseRetriever):
    """
    ChromaDB-based retriever implementation with reranking support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Chroma retriever.
        
        Args:
            config: Retriever configuration dictionary
        """
        super().__init__(config)
        self.vector_store_path = Path(config.get("vector_store_path", "data/vector_store"))
        self.embedding_model = config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_reranking = config.get("use_reranking", True)
        self.reranker = Reranker(config) if self.use_reranking else None
        logger.info(f"Using device: {self.device}")
        
    def initialize(self) -> None:
        """
        Initialize the Chroma retriever and reranker if enabled.
        """
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device}
            )
            
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=str(self.vector_store_path),
                embedding_function=self.embeddings
            )
            
            # Verify the vector store has documents
            if self.vector_store._collection.count() == 0:
                logger.warning("Vector store exists but contains no documents")
                self._is_initialized = False
                return
                
            # Initialize reranker if enabled
            if self.use_reranking and self.reranker:
                self.reranker.initialize()
                
            self._is_initialized = True
            logger.info(f"Initialized Chroma retriever with model: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma retriever: {e}")
            raise
            
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query with optional reranking.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.is_initialized():
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
            
        try:
            # First stage: Vector similarity search
            # Retrieve more documents if reranking is enabled
            initial_k = k * 2 if self.use_reranking else k
            documents = self.vector_store.as_retriever(
                search_kwargs={"k": initial_k}
            ).invoke(query)
            
            # Second stage: Reranking if enabled
            if self.use_reranking and self.reranker and self.reranker.is_initialized():
                documents = self.reranker.rerank(query, documents, top_k=k)
                
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise 