from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from langchain.schema import Document
from .retriever import BaseRetriever, ChromaRetriever
from models.base import BaseModel

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) pipeline implementation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        retriever: Optional[BaseRetriever] = None,
        model: Optional[BaseModel] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Pipeline configuration dictionary
            retriever: Retriever instance (optional)
            model: Language model instance (optional)
        """
        self.config = config
        self.retriever = retriever or ChromaRetriever(config)
        self.model = model
        
    def initialize(self) -> None:
        """
        Initialize the RAG pipeline components.
        """
        try:
            # Initialize retriever
            self.retriever.initialize()
            
            # Initialize model if provided
            if self.model:
                self.model.initialize()
                
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
            
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: The question to process
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if not self.retriever.is_initialized():
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
            
        if not self.model or not self.model.is_initialized():
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Retrieve relevant documents
            documents = self.retriever.retrieve(question, k=self.config["k"])
            
            # Construct the prompt template with better formatted context
            context_parts = []
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Documento {i} (Fonte: {source}):\n{doc.page_content}\n")
            
            context = "\n".join(context_parts)
            
            user_message = f"""Contexto:
{context}

Pergunta: {question}"""

            # Generate answer using the model
            answer = self.model.generate(user_message)
            
            # Extract source information
            sources = [
                doc.metadata.get("source", "Unknown")
                for doc in documents
            ]
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
            
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not self.retriever.is_initialized():
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
            
        try:
            if isinstance(self.retriever, ChromaRetriever):
                self.retriever.vector_store.add_documents(documents)
            else:
                raise NotImplementedError("Document addition not implemented for this retriever type")
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        return {
            "retriever": self.retriever.get_config(),
            "model": self.model.get_config() if self.model else None
        } 