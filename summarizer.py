from dataclasses import dataclass
from typing import Optional, List, Dict
from pandas import DataFrame
from summarizer_utils import SummarizerUtils
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

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

    def process_character_chunk(self, 
                              main_logs: pd.DataFrame, 
                              char_logs: pd.DataFrame, 
                              chunk_size: int = 200) -> str:
        """Process a chunk of character interactions."""
        try:
            # Ensure datetime column exists and is properly formatted
            for df in [main_logs, char_logs]:
                if 'datetime' not in df.columns:
                    df['datetime'] = pd.to_datetime(
                        df['date'], 
                        format='%d-%m-%Y %H:%M:%S',
                        errors='coerce'
                    )
            
            # Format logs with character names and limit the text size
            formatted_logs = []
            combined_logs = pd.concat([main_logs, char_logs])
            combined_logs = combined_logs.sort_values('datetime', na_position='last').head(chunk_size)
            
            total_chars = 0
            max_chars = 1000  # Limit total characters to stay within context window
            
            for _, log in combined_logs.iterrows():
                if pd.notna(log['date']) and pd.notna(log['say']):
                    log_entry = f"[{log['date']}] {log.get('character', 'Unknown')}: {log['say']}"
                    if total_chars + len(log_entry) > max_chars:
                        break
                    formatted_logs.append(log_entry)
                    total_chars += len(log_entry)
            
            if not formatted_logs:
                return ""
            
            logs_text = "\n".join(formatted_logs)
            
            # Shorter, more focused prompt
            chunk_prompt = f"""[INST] Resuma brevemente em português estas interações:

{logs_text}

Resumo: [/INST]"""

            response = self.summarizer_utils.llm(
                chunk_prompt,
                max_tokens=200,  # Reduced max tokens
                temperature=0.7,
                top_p=0.9,
                stop=["[/INST]"]
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return ""

    def summarize_character_interactions(self, 
                                       main_character: str, 
                                       date: str, 
                                       max_distance: float = 10.0,
                                       chunk_size: int = 200) -> Optional[str]:
        try:
            print(f"Buscando personagens próximos a {main_character}...")
            nearby_chars = self.summarizer_utils.find_nearby_characters(main_character, date, max_distance)
            if not nearby_chars:
                return f"Nenhuma interação próxima encontrada para {main_character} na data {date}."
            
            print(f"Encontrados {len(nearby_chars)} personagens próximos.")
            
            # Get and prepare main character logs
            main_logs = self.summarizer_utils.load_character_logs_by_date(main_character, date)
            if not isinstance(main_logs, pd.DataFrame):
                return f"Erro: Não foi possível carregar os logs de {main_character}"
            
            main_logs['character'] = main_character
            
            # Process each character's interactions in parallel
            valid_summaries = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for char in nearby_chars:
                    print(f"Processando interações com {char}...")
                    char_logs = self.summarizer_utils.load_character_logs_by_date(char, date)
                    
                    if not isinstance(char_logs, pd.DataFrame) or char_logs.empty:
                        print(f"Pulando {char} - logs não encontrados")
                        continue
                    
                    char_logs['character'] = char
                    
                    # Process smaller chunks
                    for i in range(0, len(char_logs), chunk_size):
                        char_chunk = char_logs.iloc[i:i + chunk_size].copy()
                        main_chunk = main_logs.iloc[0:min(chunk_size, len(main_logs))].copy()
                        
                        future = executor.submit(
                            self.process_character_chunk,
                            main_chunk,
                            char_chunk,
                            chunk_size
                        )
                        futures.append(future)
                
                # Collect valid results
                for future in futures:
                    result = future.result()
                    if result and len(result.strip()) > 0:
                        valid_summaries.append(result)
            
            if not valid_summaries:
                return f"Não foi possível gerar resumos válidos para as interações de {main_character}"
            
            print(f"Gerando resumo final de {len(valid_summaries)} interações válidas...")
            
            # Combine summaries in smaller groups if needed
            final_summaries = []
            for i in range(0, len(valid_summaries), 5):  # Process 5 summaries at a time
                group = valid_summaries[i:i+5]
                group_prompt = f"""[INST] Combine estes resumos em um único resumo coeso em português:

{"\n\n".join(group)}

Resumo: [/INST]"""

                group_response = self.summarizer_utils.llm(
                    group_prompt,
                    max_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["[/INST]"]
                )
                final_summaries.append(group_response['choices'][0]['text'].strip())
            
            # Final combination of all group summaries
            final_prompt = f"""[INST] Resuma as principais interações entre {main_character} e outros personagens:

{"\n\n".join(final_summaries)}

Resumo final: [/INST]"""

            final_response = self.summarizer_utils.llm(
                final_prompt,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9,
                stop=["[/INST]"]
            )
            
            return final_response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Error in summarize_character_interactions: {str(e)}")
            return str(e)