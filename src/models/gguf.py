from typing import Any, Dict, Optional
import logging
from pathlib import Path
from llama_cpp import Llama
from .base import BaseModel

logger = logging.getLogger(__name__)

class GGUFModel(BaseModel):
    """
    GGUF model implementation using llama-cpp-python.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GGUF model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model_path = Path(config.get("model_path", "model/Llama-3.1-Tulu-3-8B.gguf"))
        
    def initialize(self) -> None:
        """
        Initialize the GGUF model.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_threads=4
            )
            self._is_initialized = True
            logger.info(f"Initialized GGUF model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize GGUF model: {e}")
            raise
            
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the GGUF model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
            
            response = self._model(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["</s>", "###"],
                echo=False
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise 