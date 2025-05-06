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
        self.model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
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
            
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Prepare inputs with system message
            system_message = """Você é um assistente virtual do CNJ (Conselho Nacional de Justiça), especializado em atendimento ao usuário. 
Sua função é fornecer orientações claras e personalizadas em português, garantindo uma comunicação de qualidade.

REGRAS IMPORTANTES:
1. Responda EXCLUSIVAMENTE em português - NUNCA use palavras em inglês ou em qualquer outro idioma diferente do português
2. Seja claro, objetivo e profissional
3. Baseie sua resposta APENAS no contexto fornecido
4. Se a informação não estiver no contexto, Responda EXCLUSIVAMENTE em português que não possui informações suficientes
5. Mantenha um tom cordial e acessível
6. Priorize a segurança das informações pessoais
7. Use sempre termos em português, mesmo para conceitos técnicos
8. Evite qualquer mistura de idiomas na resposta

ESCOPOS DE ATENDIMENTO:
- Suporte Técnico: Acesso ao Portal, navegação e resolução de problemas técnicos
- Orientações: Esclarecimentos sobre serviços do Portal (processos, peticionamento, comunicações)
- Acompanhamento de processos: Auxílio no entendimento do andamento processual
- Educação e Capacitação: Tutoriais e respostas a perguntas frequentes
- Denúncias: Registro de violações ou tentativas de fraude"""
            
            # Format the prompt with the system message
            formatted_prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
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
            
            # Decode and clean up the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part after the last [/INST]
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise 