from typing import Any, Dict, Optional
import logging
import openai
from .base import BaseModel

logger = logging.getLogger(__name__)

class OpenAIModel(BaseModel):
    """
    OpenAI model implementation using OpenAI's API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.model_name = config.get("model_name", "gpt-4.1-nano-2025-04-14")
        self.client = None
        
    def initialize(self) -> None:
        """
        Initialize the OpenAI client.
        """
        try:
            logger.info(f"Initializing OpenAI client with model {self.model_name}")
            self.client = openai.OpenAI(api_key=self.api_key)
            self._is_initialized = True
            logger.info("Successfully initialized OpenAI client")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def query(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for a query with optional context.
        
        Args:
            question: The question to answer
            context: Optional context to help answer the question
            
        Returns:
            Dictionary containing the answer and sources
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            system_message = """Você é um assistente especializado em analisar logs de Ultima Online.
Sua tarefa é responder perguntas sobre as atividades e interações dos personagens de forma clara e concisa.

REGRAS:
1. Responda em português
2. Seja direto e objetivo
3. Baseie suas respostas apenas no contexto fornecido
4. Se não houver informação suficiente no contexto, indique isso claramente
5. Cite as fontes de informação quando relevante"""

            # Prepare the prompt with context if available
            if context:
                prompt = f"Contexto:\n{context}\n\nPergunta: {question}"
            else:
                prompt = question

            # Create messages for the chat completion
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 1024)
            )
            
            # Extract the generated text
            answer = response.choices[0].message.content.strip()
            
            # Return in the expected format
            return {
                "answer": answer,
                "sources": []  # OpenAI doesn't provide source tracking
            }
            
        except Exception as e:
            logger.error(f"Error generating query response: {e}")
            raise

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text between languages.
        
        Args:
            text: The text to translate
            source_lang: Source language code (e.g., 'en', 'pt')
            target_lang: Target language code (e.g., 'en', 'pt')
            
        Returns:
            Translated text
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            system_message = f"""You are a professional translator. Your task is to translate text from {source_lang} to {target_lang}.
Follow these rules:
1. Maintain the original meaning and context
2. Use natural language in the target language
3. Preserve any special formatting or symbols
4. Keep proper names unchanged
5. Translate only the text provided, nothing more
6. Respond with ONLY the translated text, no explanations or additional content"""

            # Create messages for the chat completion
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=self.config.get("max_tokens", 1024)
            )
            
            # Extract and return the translated text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
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

            # Get generation parameters
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1024))
            
            # Create messages for the chat completion
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract and return the generated text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        try:
            self.client = None
            self._is_initialized = False
            logger.info("Cleaned up OpenAI client")
            
        except Exception as e:
            logger.error(f"Error cleaning up OpenAI client: {e}")
            raise 