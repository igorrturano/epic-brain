import os
from pathlib import Path
from typing import Dict, Any
import logging
import sys
from datetime import datetime, timedelta

# Base directory setup
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Log processing settings
LOG_CONFIG = {
    "max_days_old": 5,  # Only process logs from the last 5 days
    "log_pattern": "raw/players/*/*.log",  # Pattern to match log files
    "date_format": "%d-%m-%Y",  # Format of dates in log file paths
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
        BASE_DIR / "logs"
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
            logging.FileHandler(BASE_DIR / "logs" / "app.log", encoding='utf-8'),
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
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "title": "Epic Brain Chatbot API",
    "description": "API for UO Epic Shard",
    "version": "1.0.0"
}

# RAG Pipeline settings
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "k": 4,
    "score_threshold": 0.7,
    "vector_store_path": str(VECTOR_STORE_DIR),
    "use_reranking": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "reranker_top_k": 4,
    "initial_retrieval_multiplier": 2  # Number of documents to retrieve before reranking
}

# Model settings
MODEL_CONFIG = {
    "type": "transformers", #ollama, gguf
    "model_name": "maritaca-ai/sabia-7b",
    "temperature": 0.7,
    "max_tokens": 2048,
    "use_4bit": True,  # Enable 4-bit quantization for memory efficiency
    "device": "auto"   # Will use CUDA if available
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
            "filename": str(BASE_DIR / "logs" / "app.log"),
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