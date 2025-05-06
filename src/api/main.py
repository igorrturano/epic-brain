from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
from pathlib import Path
import logging
import logging.config
from contextlib import asynccontextmanager

from config import get_config, LOGGING_CONFIG, RAW_DATA_DIR, VECTOR_STORE_DIR
from models import GGUFModel, OllamaModel, TransformersModel
from rag.pipeline import RAGPipeline
from rag.retriever import ChromaRetriever
from ingestion_pipeline import IngestionPipeline

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize RAG pipeline
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    """
    global rag_pipeline
    
    try:
        # Initialize model based on configuration
        if config["model"]["type"] == "gguf":
            model = GGUFModel(config["model"])
        elif config["model"]["type"] == "ollama":
            model = OllamaModel(config["model"])
        elif config["model"]["type"] == "transformers":
            model = TransformersModel(config["model"])
        else:
            raise ValueError(f"Unsupported model type: {config['model']['type']}")
            
        # Initialize retriever
        retriever = ChromaRetriever(config["rag"])
            
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            config=config["rag"],
            retriever=retriever,
            model=model
        )
        
        rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

# Initialize FastAPI app
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
for directory in [RAW_DATA_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize components
logger.info("Initializing ingestion pipeline...")
ingestion_pipeline = IngestionPipeline(
    raw_directory=str(RAW_DATA_DIR),
    vector_store_path=str(VECTOR_STORE_DIR)
)

class QueryRequest(BaseModel):
    question: str
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the documents using the RAG pipeline.
    
    Args:
        request: Query request containing the question
        
    Returns:
        Query response containing the answer and sources
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
    try:
        # Update max_results in config
        config["rag"]["k"] = request.max_results
        
        # Process query
        result = rag_pipeline.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get statistics about the RAG pipeline.
    
    Returns:
        Pipeline statistics
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
    try:
        return rag_pipeline.get_stats()
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 