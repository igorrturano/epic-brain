from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
import logging.config
from contextlib import asynccontextmanager
import asyncio
from functools import partial

from config import get_config, LOGGING_CONFIG, RAW_DATA_DIR, VECTOR_STORE_DIR
from models import OpenAIModel
from rag.pipeline import RAGPipeline
from rag.retriever import ChromaRetriever
from ingestion_pipeline import IngestionPipeline

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize components
rag_pipeline = None
model = None
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    """
    global rag_pipeline, model, retriever
    
    try:
        # Initialize model
        model = OpenAIModel(config["model"])
        model.initialize()
        
        # Initialize retriever
        retriever = ChromaRetriever(config["rag"])
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            config=config["rag"],
            retriever=retriever,
            model=model
        )
        
        rag_pipeline.initialize()
        logger.info("Resources initialized successfully")
        
        yield
        
        # Cleanup
        if rag_pipeline:
            rag_pipeline.cleanup()
        if model:
            model.cleanup()
        if retriever:
            retriever.cleanup()
            
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
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

# Initialize ingestion pipeline
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

class TranslationRequest(BaseModel):
    target: str
    origin: str
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the documents using the RAG pipeline.
    
    Args:
        request: Query request containing the question
        
    Returns:
        Query response containing the answer and sources
    """
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=500,
                detail="RAG pipeline not initialized"
            )
            
        # Update max_results in config
        config["rag"]["k"] = request.max_results
        
        # Run query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(rag_pipeline.query, request.question)
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest) -> TranslationResponse:
    """
    Translate text between languages using the model.
    
    Args:
        request: Translation request containing source text and languages
        
    Returns:
        Translation response containing the translated text
    """
    try:
        if not model:
            raise HTTPException(
                status_code=500,
                detail="Model not initialized"
            )
            
        # Run translation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        translated_text = await loop.run_in_executor(
            None,
            partial(
                model.translate,
                text=request.text,
                source_lang=request.origin,
                target_lang=request.target
            )
        )
        
        return TranslationResponse(
            translated_text=translated_text
        )
        
    except Exception as e:
        logger.error(f"Error processing translation: {e}")
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