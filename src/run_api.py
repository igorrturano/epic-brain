import uvicorn
from pathlib import Path
import logging
import logging.config
import sys
from os.path import dirname, abspath

# Add src directory to Python path
src_dir = dirname(dirname(abspath(__file__)))
sys.path.append(src_dir)

from config import get_config, LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

if __name__ == "__main__":
    try:
        logger.info("Starting API server...")
        uvicorn.run(
            "api.main:app",
            host=config["api"]["host"],
            port=config["api"]["port"],
            reload=config["api"]["debug"]
        )
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        raise 