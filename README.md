# Epic Brain

**Epic Brain** is a bot developed to help GMs (Game Masters) track the interactions and actions of characters on [epic-shard.com](https://epic-shard.com). It processes logs of actions and dialogues of the characters and generates interpretative summaries, making it easier to follow the players' progress.

## Features

- **Log Processing**: Automatically processes and cleans character logs from the game server
- **RAG Pipeline**: Uses Retrieval-Augmented Generation for accurate and contextual responses
- **FastAPI Backend**: Provides a RESTful API for log processing and querying
- **Local Execution**: Everything is executed locally without the need for external APIs
- **Multiple Model Support**: Supports both GGUF and Transformers models
- **Vector Store**: Maintains a vector database for efficient log retrieval and querying

## Technologies Used

- **Python 3.12**: The main programming language of the project
- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications powered by language models
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **Transformers**: Library for state-of-the-art NLP models
- **PyTorch**: Deep learning framework for model inference
- **Sentence Transformers**: For generating embeddings
- **Pandas**: For data manipulation and processing

## How to Set Up

### Prerequisites

1. **Python 3.12 or higher**
2. **VPS or local machine** with sufficient memory to run the models
3. **.env File**: Create a `.env` file at the root of the project with the following structure:
   ```env
   # Directories
   DATA_DIR=/path/to/data
   RAW_DATA_DIR=/path/to/raw
   VECTOR_STORE_DIR=/path/to/vector_store
   LOGS_DIR=/path/to/game/logs

   # API Settings
   API_HOST=0.0.0.0
   API_PORT=8000
   API_DEBUG=false
   API_TITLE="Epic Brain Chatbot API"
   API_DESCRIPTION="API for UO Epic Shard"
   API_VERSION=1.0.0
   API_RATE_LIMIT=1
   API_RATE_LIMIT_WINDOW=600
   API_MAX_CONCURRENT_REQUESTS=1

   # Model Settings
   MODEL_TYPE=transformers
   MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
   MODEL_TEMPERATURE=0.3
   MODEL_MAX_TOKENS=1024
   MODEL_USE_4BIT=true
   MODEL_DEVICE=auto

   # RAG Settings
   RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   RAG_K=4
   RAG_SCORE_THRESHOLD=0.7
   RAG_USE_RERANKING=true
   RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   RAG_RERANKER_TOP_K=4
   RAG_INITIAL_RETRIEVAL_MULTIPLIER=2

   # Log Settings
   LOG_MAX_DAYS_OLD=5
   LOG_PATTERN=*.log
   LOG_DATE_FORMAT=%d-%m-%Y
   ```

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/epic-brain.git
   cd epic-brain
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Run the API server**:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

## Project Structure

```
epic-brain/
├── src/
│   ├── api.py              # FastAPI application
│   ├── config.py           # Configuration settings
│   ├── clean_logs.py       # Log processing and cleaning
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py     # RAG pipeline implementation
│   │   └── utils.py        # RAG utility functions
│   └── models/
│       ├── __init__.py
│       └── llm.py          # Language model interface
├── data/
│   ├── raw/               # Raw processed logs
│   └── vector_store/      # Vector database storage
├── pyproject.toml         # Project dependencies and metadata
├── .env                   # Environment configuration
└── README.md             # This file
```

## API Endpoints

- `GET /stats`: Get statistics about the RAG pipeline.
- `POST /query`: Query the processed logs using natural language
- `GET /health`: Health check endpoint
- `POST /translate`: Translate a message

## Performance Notes

The system is optimized for:
- Efficient log processing and cleaning
- Fast vector search and retrieval
- Low memory footprint with 4-bit quantization
- Scalable API architecture

Processing time depends on:
- Size of log files
- Number of concurrent requests
- Hardware specifications
