from dataclasses import dataclass
from typing import Optional
from pandas import DataFrame
from summarizer_utils import SummarizerUtils

@dataclass
class SummarizerConfig:
    summary_prompt: str

class Summarizer:
    def __init__(self):
        self.config = self._initialize_config()
        self.summarizer_utils = SummarizerUtils()
    
    def _initialize_config(self) -> SummarizerConfig:
        return SummarizerConfig(
            summary_prompt=self._load_prompt_template()
        )
    
    def _load_prompt_template(self) -> str:
        try:
            with open('summarization_prompt.txt', 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            print("Warning: summarization_prompt.txt not found, using default prompt")
            return """Você é um assistente que fala português..."""
    
    def create_chat_template(self, prompt: str) -> str:
        return f"""[INST] {prompt} [/INST]"""
    
    def summarize_logs(self, character_name: str, logs_df: DataFrame, 
                      max_tokens: int = 1000) -> Optional[str]:
        if logs_df.empty:
            return "Nenhum log encontrado para sumarização."

        try:
            formatted_logs = [
                f"[{row['date']}] {row['say']}" 
                for _, row in logs_df.iterrows()
            ]
            
            logs_text = "\n".join(formatted_logs)
            prompt = self.config.summary_prompt.format(
                character_name=character_name,
                logs=logs_text
            )
            formatted_prompt = self.create_chat_template(prompt)

            response = self.summarizer_utils.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                stop=["[/INST]", "\n\n"]
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error in summarize_logs: {e}")
            return str(e)

    def summarize_logs_by_date(self, character_name: str, date: str) -> Optional[str]:
        logs_df = self.summarizer_utils.load_character_logs_by_date(character_name, date)
        if isinstance(logs_df, str) and not logs_df:
            return "Nenhum log encontrado para a data especificada."
            
        return self.summarize_logs(character_name, logs_df)
    
    def summarize_all_logs(self, character_name: str) -> Optional[str]:
        logs_df = self.summarizer_utils.load_character_logs(character_name)
        if isinstance(logs_df, str) and not logs_df:
            return "Nenhum log encontrado para o personagem."
            
        return self.summarize_logs(character_name, logs_df)

    def summarize_character_interactions(self, main_character: str, date: str, max_distance: float = 10.0) -> Optional[str]:
        try:
            # Find nearby characters
            nearby_chars = self.summarizer_utils.find_nearby_characters(main_character, date, max_distance)
            if not nearby_chars:
                return f"Nenhuma interação próxima encontrada para {main_character} na data {date}."
            
            # Get logs for all characters
            all_logs = []
            all_logs.extend(self.summarizer_utils.load_character_logs_by_date(main_character, date).to_dict('records'))
            
            for char in nearby_chars:
                char_logs = self.summarizer_utils.load_character_logs_by_date(char, date).to_dict('records')
                all_logs.extend(char_logs)
                
            # Sort logs by timestamp
            all_logs.sort(key=lambda x: x['date'])
            
            # Format logs with character names
            formatted_logs = [
                f"[{log['date']}] {log['character']}: {log['say']}" 
                for log in all_logs
            ]
            
            logs_text = "\n".join(formatted_logs)
            
            # Create a special prompt for interaction analysis
            interaction_prompt = f"""[INST] Você é um assistente especializado em análise de interações entre personagens. Analise as interações entre {main_character} e os personagens próximos ({', '.join(nearby_chars)}).

1. **SEMPRE responda em português**
2. Identifique os principais momentos de interação
3. Analise:
   - O contexto das interações
   - O tom das conversas
   - Ações e reações dos personagens
   - Possíveis relacionamentos ou alianças formadas

Logs das interações:
{logs_text}

Forneça um resumo detalhado das interações: [/INST]"""

            response = self.summarizer_utils.llm(
                interaction_prompt,
                max_tokens=2000,
                temperature=0.6,
                top_p=0.9,
                stop=["[/INST]", "\n\n"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Error in summarize_character_interactions: {e}")
            return str(e)