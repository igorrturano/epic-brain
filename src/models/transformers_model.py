from typing import Any, Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import BaseModel

logger = logging.getLogger(__name__)

class TransformersModel(BaseModel):
    """
    Transformers model implementation using HuggingFace's transformers library.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Transformers model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantization_config = None
        
        # Setup quantization if specified and CUDA is available
        if config.get("use_4bit", True) and torch.cuda.is_available():
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif config.get("use_4bit", True) and not torch.cuda.is_available():
            logger.warning("4-bit quantization requested but CUDA not available. Running in full precision mode.")
        
    def initialize(self) -> None:
        """
        Initialize the Transformers model.
        """
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with appropriate configuration
            model_kwargs = {
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # Enable low CPU memory usage
                "torch_dtype": torch.float32 if not torch.cuda.is_available() else torch.float16
            }
            
            # Add quantization config only if CUDA is available
            if self.quantization_config and torch.cuda.is_available():
                model_kwargs["quantization_config"] = self.quantization_config
            
            try:
                # Load model
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Move model to CPU if CUDA is not available
                if not torch.cuda.is_available():
                    self._model = self._model.to(self.device)
                
                self._is_initialized = True
                logger.info(f"Successfully initialized model: {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                if "out of memory" in str(e).lower():
                    logger.error("Out of memory error. Consider using a smaller model or increasing swap space.")
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise
            
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            system_message: Optional custom system message. If None, uses default.
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Use default system message if none provided
            if system_message is None:
                system_message = """Você é um assistente especializado em analisar logs de Ultima Online.
Sua tarefa é resumir as atividades e interações dos personagens de forma clara e concisa.

REGRAS:
1. Responda em português
2. Seja direto e objetivo
3. Foque nos eventos mais relevantes
4. Inclua localizações importantes
5. Mencione outros personagens relevantes
6. Mantenha a cronologia dos eventos

FORMATO DA RESPOSTA:
Resumo das atividades do personagem:
[Resumo conciso das principais atividades]

Eventos principais:
- [Lista de eventos relevantes com localizações]

Observações: [Comportamento ou padrões observados]"""

            # Format conversation using Gemma's format
            formatted_prompt = f"<start_of_turn>user\n{system_message}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            # Tokenize the formatted prompt
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)
            
            # Get generation parameters
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
            
            # Generate
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0])
            
            # Clean up the response by removing the prompt and any template markers
            response = response.replace(formatted_prompt, "")
            if self.config.get("stop_word", "<end_of_turn>") in response:
                response = response.split(self.config.get("stop_word", "<end_of_turn>"))[0]
            
            # Remove special tokens
            special_tokens = ["<bos>", "<eos>", "<pad>", "<unk>"]
            for token in special_tokens:
                response = response.replace(token, "")
            
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise 

    def cleanup(self) -> None:
        """
        Clean up model resources and release GPU memory.
        """
        try:
            if hasattr(self, '_model'):
                # Move model to CPU
                self._model.to('cpu')
                # Delete model
                del self._model
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Cleaned up model: {self.model_name}")
            
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                
            self._is_initialized = False
            
        except Exception as e:
            logger.error(f"Error cleaning up model: {e}")
            raise 