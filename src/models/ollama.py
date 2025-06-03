from typing import Any, Dict, Optional
import logging
import requests
from requests.exceptions import RequestException
from .base import BaseModel

logger = logging.getLogger(__name__)

class OllamaModel(BaseModel):
    """
    Ollama model implementation using the Ollama API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ollama model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "mistral")
        self.timeout = config.get("timeout", 30)
        self.retry_attempts = config.get("retry_attempts", 3)
        
    def initialize(self) -> None:
        """
        Initialize the Ollama model.
        """
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            self._is_initialized = True
            logger.info(f"Initialized Ollama model: {self.model_name}")
        except RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise RuntimeError(f"Could not connect to Ollama server: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model: {e}")
            raise
            
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stream": False,
                        "options": {
                            "stop": ["[INST]", "[/INST]"],  # Stop generation at these tokens
                            "system": "Você é um assistente virtual do Epic Brain (Shard de ULtima Online), especializado em atendimento ao usuário. Sua função é fornecer orientações claras e personalizadas em português, garantindo uma comunicação de qualidade."
                        }
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["response"].strip()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.error(f"Model {self.model_name} not found on Ollama server")
                    raise RuntimeError(f"Model {self.model_name} not found on Ollama server")
                elif e.response.status_code == 500:
                    logger.warning(f"Server error on attempt {attempt + 1}/{self.retry_attempts}")
                    if attempt == self.retry_attempts - 1:
                        raise RuntimeError("Ollama server error after multiple attempts")
                    continue
                else:
                    raise
            except RequestException as e:
                logger.error(f"Network error during generation: {e}")
                if attempt == self.retry_attempts - 1:
                    raise RuntimeError("Failed to generate response after multiple attempts")
                continue
            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}")
                raise 