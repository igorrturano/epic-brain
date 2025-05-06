from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all language models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self._model = None
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model. Must be implemented by subclasses.
        """
        pass
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model. Must be implemented by subclasses.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
        
    def is_initialized(self) -> bool:
        """
        Check if the model is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._is_initialized
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return self.config.copy()
        
    def __str__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(config={self.config})" 