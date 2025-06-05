import os
from pathlib import Path
from typing import Dict, Any
import logging
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Base directory setup
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(BASE_DIR / "raw")))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(DATA_DIR / "vector_store")))
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))

# Log processing settings
LOG_CONFIG = {
    "max_days_old": int(os.getenv("LOG_MAX_DAYS_OLD", "5")),
    "log_pattern": os.getenv("LOG_PATTERN", "*.log"),
    "date_format": os.getenv("LOG_DATE_FORMAT", "%d-%m-%Y"),
}

# Model parameters
MODEL_PARAMS = {
    "n_ctx": 2048,
    "n_threads": 8,
    "n_gpu_layers": 50,
    "temperature": 0.1
}

# Create necessary directories
def create_directories():
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        VECTOR_STORE_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Logging configuration
def setup_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / "app.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure console handler to use utf-8
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)
    
    return logging.getLogger(__name__)

# API settings
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true",
    "title": os.getenv("API_TITLE", "Epic Brain Chatbot API"),
    "description": os.getenv("API_DESCRIPTION", "API for UO Epic Shard"),
    "version": os.getenv("API_VERSION", "1.0.0"),
    "rate_limit": int(os.getenv("API_RATE_LIMIT", "1")),
    "rate_limit_window": int(os.getenv("API_RATE_LIMIT_WINDOW", "600")),
    "max_concurrent_requests": int(os.getenv("API_MAX_CONCURRENT_REQUESTS", "1"))
}

# RAG Pipeline settings
RAG_CONFIG = {
    "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "k": int(os.getenv("RAG_K", "4")),
    "score_threshold": float(os.getenv("RAG_SCORE_THRESHOLD", "0.7")),
    "vector_store_path": str(VECTOR_STORE_DIR),
    "use_reranking": os.getenv("RAG_USE_RERANKING", "true").lower() == "true",
    "reranker_model": os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "reranker_top_k": int(os.getenv("RAG_RERANKER_TOP_K", "4")),
    "initial_retrieval_multiplier": int(os.getenv("RAG_INITIAL_RETRIEVAL_MULTIPLIER", "2"))
}

# Model settings
MODEL_CONFIG = {
    "type": os.getenv("MODEL_TYPE", "openai"),
    "model_name": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", "1024")),
    "use_4bit": os.getenv("MODEL_USE_4BIT", "true").lower() == "true",
    "device": os.getenv("MODEL_DEVICE", "auto"),
    "stop_word": os.getenv("MODEL_STOP_WORD", "[/INST]"),
    "api_key": os.getenv("OPENAI_API_KEY", "")
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(LOGS_DIR / "app.log"),
            "level": "DEBUG",
            "encoding": "utf-8"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict containing all configuration settings
    """
    return {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "vector_store_dir": VECTOR_STORE_DIR,
        "api": API_CONFIG,
        "rag": RAG_CONFIG,
        "model": MODEL_CONFIG,
        "logging": LOGGING_CONFIG
    } 