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
        
        # Setup quantization if specified
        if config.get("use_4bit", True):
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
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
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=self.quantization_config,
                trust_remote_code=True
            )
            
            self._is_initialized = True
            logger.info(f"Successfully initialized model: {self.model_name}")
            
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
                system_message = """Você é um assistente virtual especializado em Ultima Online, focado em um shard de roleplay.
Sua função é analisar e resumir as interações e eventos que ocorrem no shard, baseando-se nos logs de chat e ações dos jogadores.

REGRAS IMPORTANTES:
1. Responda EXCLUSIVAMENTE em português
2. Agrupe eventos relacionados em categorias lógicas
3. Identifique padrões de comportamento e atividades
4. Forneça contexto sobre as localizações e interações
5. Mantenha a cronologia dos eventos quando relevante
6. Use termos apropriados ao contexto medieval/fantasia
7. Seja conciso mas informativo

INSTRUÇÕES DE RESUMO:
1. Agrupe atividades similares (ex: todas as atividades de coleta, todas as interações de combate)
2. Identifique personagens principais e suas atividades
3. Destaque eventos significativos ou mudanças importantes
4. Forneça contexto sobre localizações quando relevante
5. Mantenha um tom narrativo que faça sentido para o universo de Ultima Online

EXEMPLO DE ESTRUTURA DE RESPOSTA:
1. Visão Geral: Breve resumo dos principais eventos
2. Atividades por Categoria: Agrupamento de eventos similares
3. Personagens Principais: Ações e interações dos personagens mais ativos
4. Eventos Significativos: Destaque para momentos importantes
5. Contexto e Localização: Informações sobre onde os eventos ocorreram"""

            # Format conversation using the chat template
            conversation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Format and tokenize using the chat template
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
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
            
            # Extract only the response after [/INST]
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            
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