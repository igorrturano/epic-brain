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
                              main_character: str,
                              other_character: str,
                              main_logs: pd.DataFrame, 
                              char_logs: pd.DataFrame, 
                              chunk_size: int = 200) -> str:
        try:
            print(f"  - Processando chunk de interações entre {main_character} e {other_character}")
            
            # Ensure proper datetime format for both dataframes
            for df in [main_logs, char_logs]:
                if 'datetime' not in df.columns:
                    # Convert the date column to datetime using the correct format
                    df['datetime'] = pd.to_datetime(
                        df['date'],
                        format='%d-%m-%Y %H:%M:%S',
                        errors='coerce'
                    )
            
            # Format logs with character names and limit the text size
            formatted_logs = []
            combined_logs = pd.concat([main_logs, char_logs])
            combined_logs = combined_logs.sort_values('datetime', na_position='last').head(chunk_size)
            
            for _, log in combined_logs.iterrows():
                if pd.notna(log['date']) and pd.notna(log['say']):
                    # Format the log entry with coordinates
                    coord = log.get('coord', 'Unknown location')
                    char_name = log.get('character', 'Unknown')
                    log_entry = f"[{log['date']}] ({coord}) {char_name}: {log['say']}"
                    formatted_logs.append(log_entry)
            
            if not formatted_logs:
                print(f"    - Nenhuma interação encontrada neste chunk")
                return ""
            
            print(f"    - Encontradas {len(formatted_logs)} interações")
            logs_text = "\n".join(formatted_logs)
            
            chunk_prompt = f"""[INST] Analise e resuma em português as interações entre {main_character} e {other_character}:

{logs_text}

Forneça um breve resumo focando em:
1. O tipo de interação entre eles
2. O tom da conversa
3. Ações importantes realizadas
4. Locais onde interagiram (baseado nas coordenadas)

Resumo: [/INST]"""

            response = self.summarizer_utils.llm(
                chunk_prompt,
                max_tokens=300,
                temperature=0.7,
                top_p=0.9,
                stop=["[/INST]"]
            )
            
            result = response['choices'][0]['text'].strip()
            if result:
                print(f"    - Resumo gerado com sucesso ({len(result)} caracteres)")
            return result
        except Exception as e:
            print(f"    - Erro processando chunk: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full error trace
            return ""

    def load_and_prepare_logs(self, character_name: str, date: str) -> pd.DataFrame:
        """Helper function to load and prepare logs with correct format"""
        try:
            logs_df = self.summarizer_utils.load_character_logs_by_date(character_name, date)
            if not isinstance(logs_df, pd.DataFrame) or logs_df.empty:
                return pd.DataFrame()
            
            # Ensure all required columns exist
            logs_df['character'] = character_name
            logs_df['datetime'] = pd.to_datetime(
                logs_df['date'],
                format='%d-%m-%Y %H:%M:%S',
                errors='coerce'
            )
            
            return logs_df
        except Exception as e:
            print(f"Error preparing logs for {character_name}: {str(e)}")
            return pd.DataFrame()

    def summarize_character_interactions(self, 
                                       main_character: str, 
                                       date: str, 
                                       max_distance: float = 10.0,
                                       chunk_size: int = 200) -> Optional[str]:
        try:
            print(f"\nBuscando personagens próximos a {main_character}...")
            nearby_chars = self.summarizer_utils.find_nearby_characters(main_character, date, max_distance)
            if not nearby_chars:
                return f"Nenhuma interação próxima encontrada para {main_character} na data {date}."
            
            print(f"Encontrados {len(nearby_chars)} personagens próximos: {', '.join(nearby_chars)}\n")
            
            # Get and prepare main character logs
            main_logs = self.load_and_prepare_logs(main_character, date)
            if main_logs.empty:
                return f"Erro: Não foi possível carregar os logs de {main_character}"
            
            # Process each character's interactions in parallel
            valid_summaries = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for char in nearby_chars:
                    print(f"\nAnalisando interações com {char}...")
                    char_logs = self.load_and_prepare_logs(char, date)
                    
                    if char_logs.empty:
                        print(f"  - Pulando {char} - logs não encontrados")
                        continue
                    
                    # Process smaller chunks
                    for i in range(0, len(char_logs), chunk_size):
                        char_chunk = char_logs.iloc[i:i + chunk_size].copy()
                        main_chunk = main_logs.iloc[0:min(chunk_size, len(main_logs))].copy()
                        
                        future = executor.submit(
                            self.process_character_chunk,
                            main_character,
                            char,
                            main_chunk,
                            char_chunk,
                            chunk_size
                        )
                        futures.append((char, future))
                
                # Collect valid results
                for char, future in futures:
                    result = future.result()
                    if result and len(result.strip()) > 0:
                        print(f"  - Resumo válido gerado para interação com {char}")
                        valid_summaries.append(f"Interações com {char}:\n{result}")
            
            if not valid_summaries:
                return f"Não foi possível gerar resumos válidos para as interações de {main_character}"
            
            print(f"\nGerando resumo final de {len(valid_summaries)} interações válidas...")
            
            final_prompt = f"""[INST] Combine os seguintes resumos em uma narrativa coesa em português, descrevendo as interações de {main_character} com outros personagens:

{"\n\n".join(valid_summaries)}

Forneça um resumo geral que destaque:
1. Os principais personagens com quem {main_character} interagiu
2. O tipo de interações mais frequentes
3. Momentos importantes ou padrões de comportamento

Resumo final: [/INST]"""

            final_response = self.summarizer_utils.llm(
                final_prompt,
                max_tokens=800,
                temperature=0.6,
                top_p=0.9,
                stop=["[/INST]"]
            )
            
            result = final_response['choices'][0]['text'].strip()
            print("\nResumo final gerado com sucesso!")
            return result
            
        except Exception as e:
            print(f"\nErro em summarize_character_interactions: {str(e)}")
            return str(e)