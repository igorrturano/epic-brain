from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
from pathlib import Path
import logging
import logging.config
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

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

# Rate limiting configuration
request_timestamps = defaultdict(list)
request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(config["api"]["max_concurrent_requests"])

async def process_queue():
    """Process requests in the queue."""
    while True:
        try:
            request_data = await request_queue.get()
            try:
                # Run synchronous function in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: request_data["func"](*request_data["args"], **request_data["kwargs"])
                )
                request_data["future"].set_result(result)
            except Exception as e:
                request_data["future"].set_exception(e)
            finally:
                request_queue.task_done()
        except Exception as e:
            logger.error(f"Error processing queue: {e}")

async def check_rate_limit(client_id: str) -> bool:
    """Check if the client has exceeded the rate limit."""
    now = datetime.now()
    window_start = now - timedelta(seconds=config["api"]["rate_limit_window"])
    
    # Clean old timestamps
    request_timestamps[client_id] = [ts for ts in request_timestamps[client_id] if ts > window_start]
    
    if len(request_timestamps[client_id]) >= config["api"]["rate_limit"]:
        return False
    
    request_timestamps[client_id].append(now)
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    """
    global rag_pipeline
    
    try:
        # Start queue processor
        asyncio.create_task(process_queue())
        
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
async def query_documents(request: QueryRequest, req: Request) -> QueryResponse:
    """
    Query the documents using the RAG pipeline.
    
    Args:
        request: Query request containing the question
        req: FastAPI request object for client identification
        
    Returns:
        Query response containing the answer and sources
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    # Get client identifier (IP address)
    client_id = req.client.host
    
    # Check rate limit
    if not await check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    try:
        # Create a future for the result
        future = asyncio.Future()
        
        # Update max_results in config
        config["rag"]["k"] = request.max_results
        
        # Add request to queue
        await request_queue.put({
            "func": rag_pipeline.query,
            "args": [request.question],
            "kwargs": {},
            "future": future
        })
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=600)  # seconds timeout
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Request timed out. Please try again."
            )
        
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