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

# Initialize components
rag_pipeline = None
model = None
retriever = None
is_initialized = False
initialization_lock = asyncio.Lock()

# Rate limiting configuration
request_timestamps = defaultdict(list)
request_queue = asyncio.Queue()
semaphore = asyncio.Semaphore(config["api"]["max_concurrent_requests"])

async def initialize_resources():
    """Initialize RAG pipeline and model resources."""
    global rag_pipeline, model, retriever, is_initialized
    
    if is_initialized:
        return
        
    async with initialization_lock:
        if is_initialized:  # Double check after acquiring lock
            return
            
        try:
            logger.info("Initializing resources...")
            
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
            is_initialized = True
            logger.info("Resources initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize resources: {e}")
            raise

async def cleanup_resources():
    """Clean up resources when they're no longer needed."""
    global rag_pipeline, model, retriever, is_initialized
    
    if not is_initialized:
        return
        
    async with initialization_lock:
        if not is_initialized:  # Double check after acquiring lock
            return
            
        try:
            logger.info("Cleaning up resources...")
            
            # Clean up resources in reverse order of initialization
            if rag_pipeline:
                # Call cleanup method if it exists
                if hasattr(rag_pipeline, 'cleanup'):
                    rag_pipeline.cleanup()
                rag_pipeline = None
                
            if retriever:
                # Call cleanup method if it exists
                if hasattr(retriever, 'cleanup'):
                    retriever.cleanup()
                retriever = None
                
            if model:
                # Call cleanup method if it exists
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                # Force CUDA cache clear
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    model.model.to('cpu')
                model = None
                
            # Force CUDA cache clear
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except ImportError:
                pass
                
            is_initialized = False
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            raise

async def process_queue():
    """Process requests in the queue."""
    while True:
        try:
            request_data = await request_queue.get()
            try:
                # Initialize resources if needed
                await initialize_resources()
                
                if not rag_pipeline:
                    request_data["future"].set_exception(
                        Exception("Failed to initialize RAG pipeline")
                    )
                    continue
                
                # Run synchronous function in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: request_data["func"](*request_data["args"], **request_data["kwargs"])
                )
                
                if result is None:
                    request_data["future"].set_exception(
                        Exception("Failed to process query")
                    )
                else:
                    request_data["future"].set_result(result)
                
                # Clean up resources after processing
                await cleanup_resources()
                
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

class TranslationRequest(BaseModel):
    target: str
    origin: str
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

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
        
        # Add request to queue with the question directly
        await request_queue.put({
            "func": lambda q: rag_pipeline.query(q) if rag_pipeline else None,
            "args": [request.question],
            "kwargs": {},
            "future": future
        })
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=600)  # seconds timeout
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize RAG pipeline"
                )
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
        # Create a future for the result
        future = asyncio.Future()
        
        # Create translation system message
        system_message = f"""You are a professional translator. Your task is to translate text from {request.origin} to {request.target}.
Follow these rules:
1. Maintain the original meaning and context
2. Use natural language in the target language
3. Preserve any special formatting or symbols
4. Keep proper names unchanged
5. Translate only the text provided, nothing more
6. Respond with ONLY the translated text, no explanations or additional content"""

        # Add request to queue
        await request_queue.put({
            "func": lambda q, sys_msg: model.generate(q, system_message=sys_msg) if model else None,
            "args": [request.text, system_message],
            "kwargs": {},
            "future": future
        })
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=60)  # 60 seconds timeout
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize model"
                )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Translation request timed out. Please try again."
            )
        
        return TranslationResponse(
            translated_text=result
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