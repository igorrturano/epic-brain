from dataclasses import dataclass
from typing import Optional, List
from pandas import DataFrame
from summarizer_utils import SummarizerUtils
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

@dataclass
class SummarizerConfig:
    summary_prompt: str

def approximate_token_count(text: str) -> int:
    # Rough approximation: 1 token ≈ 4 characters.
    return len(text) // 4

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
                          chunk_size: int = 50) -> str:
        try:
            print(f"  - Processando interações entre {main_character} e {other_character}")
            
            # Ensure both DataFrames have datetime column
            for df in [main_logs, char_logs]:
                if 'datetime' not in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M:%S')
            
            # Format logs with character names but limit size strictly
            formatted_logs = []
            combined_logs = pd.concat([main_logs, char_logs])
            combined_logs = combined_logs.sort_values('datetime', na_position='last').head(chunk_size)
            
            total_chars = 0
            max_chars = 2000  # Strict character limit
            
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
            
            # Simplified prompt to reduce context size
            chunk_prompt = f"""[INST] Resuma em português estas interações entre {main_character} e {other_character}:

{logs_text}

Resumo breve: [/INST]"""

            response = self.summarizer_utils.llm(
                chunk_prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                stop=["[/INST]"]
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"    - Erro: {str(e)}")
            return ""

    def summarize_character_interactions(self, 
                                   main_character: str, 
                                   date: str, 
                                   max_distance: float = 10.0) -> Optional[str]:
        try:
            print(f"\nBuscando personagens próximos a {main_character}...")
            nearby_chars = self.summarizer_utils.find_nearby_characters(main_character, date, max_distance)
            if not nearby_chars:
                return f"Nenhuma interação próxima encontrada para {main_character} na data {date}."
            
            print(f"Encontrados {len(nearby_chars)} personagens próximos.")
            
            # Get main character logs and prepare datetime
            main_logs = self.summarizer_utils.load_character_logs_by_date(main_character, date)
            if not isinstance(main_logs, pd.DataFrame):
                return f"Erro: Não foi possível carregar os logs de {main_character}"
            
            main_logs['character'] = main_character
            main_logs['datetime'] = pd.to_datetime(main_logs['date'], format='%d-%m-%Y %H:%M:%S')
            
            # Process each character's interactions
            all_summaries = []
            
            for char in nearby_chars:
                print(f"\nProcessando interações com {char}...")
                char_logs = self.summarizer_utils.load_character_logs_by_date(char, date)
                
                if not isinstance(char_logs, pd.DataFrame) or char_logs.empty:
                    print(f"  - Pulando {char} - logs não encontrados")
                    continue
                
                char_logs['character'] = char
                char_logs['datetime'] = pd.to_datetime(char_logs['date'], format='%d-%m-%Y %H:%M:%S')
                
                # Get time range for this character's logs
                start_time = char_logs['datetime'].min()
                end_time = char_logs['datetime'].max()
                
                if pd.isna(start_time) or pd.isna(end_time):
                    print(f"  - Pulando {char} - datas inválidas")
                    continue
                
                # Process in smaller time windows
                window_size = pd.Timedelta(minutes=30)
                time_windows = []
                
                current_time = start_time
                while current_time < end_time:
                    window_end = current_time + window_size
                    
                    # Get logs for this time window
                    char_window = char_logs[
                        (char_logs['datetime'] >= current_time) & 
                        (char_logs['datetime'] < window_end)
                    ]
                    main_window = main_logs[
                        (main_logs['datetime'] >= current_time) & 
                        (main_logs['datetime'] < window_end)
                    ]
                    
                    if not char_window.empty and not main_window.empty:
                        summary = self.process_character_chunk(
                            main_character,
                            char,
                            main_window,
                            char_window,
                            chunk_size=50
                        )
                        if summary:
                            time_windows.append(summary)
                
                    current_time = window_end
                
                if time_windows:
                    # Combine summaries for this character
                    char_summary = f"Interações com {char}:\n" + "\n".join(time_windows)
                    all_summaries.append(char_summary)
            
            if not all_summaries:
                return f"Não foi possível gerar resumos válidos para as interações de {main_character}"
            
            print("\nCombinando resumos finais...")
            
            # Combine all summaries into one
            final_prompt = f"""[INST] Combine estes resumos de interações em um único resumo em português:

{"\n\n".join(all_summaries)}

Resumo final das interações de {main_character}: [/INST]"""

            response = self.summarizer_utils.llm(
                final_prompt,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9,
                stop=["[/INST]"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Erro: {str(e)}")
            return str(e)

    def hierarchical_summary(self, summaries: List[str], main_character: str, max_tokens: int = 1800) -> str:
        """
        Recursively combine summaries to ensure the prompt does not exceed the context window.
        This function chunks the summaries into groups (by character count), summarizes each chunk,
        and then combines the intermediate summaries into a final summary.
        """
        # Using a rough conversion: max_tokens * 4 characters for the maximum allowed characters.
        max_chunk_chars = max_tokens * 4

        def combine_into_chunks(summaries_list: List[str]) -> List[str]:
            chunks = []
            current_chunk = ""
            for summary in summaries_list:
                addition = ("\n\n" + summary) if current_chunk else summary
                if len(current_chunk) + len(addition) > max_chunk_chars:
                    chunks.append(current_chunk)
                    current_chunk = summary
                else:
                    current_chunk += addition
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        intermediate = summaries
        iteration = 0

        while len(intermediate) > 1:
            iteration += 1
            new_intermediate = []
            chunks = combine_into_chunks(intermediate)
            print(f"Hierarchical summarization iteration {iteration}: combinando {len(intermediate)} resumos em {len(chunks)} chunks.")
            for chunk in chunks:
                prompt = f"""[INST] Analise e combine os seguintes resumos de interações em uma narrativa coesa, focando nos aspectos principais das interações de {main_character}:

{chunk}

Resumo: [/INST]"""
                if approximate_token_count(prompt) > 2016:
                    print("Aviso: o prompt pode ainda exceder o limite de contexto. Considere reduzir o tamanho do chunk.")
                
                response = self.summarizer_utils.llm(
                    prompt,
                    max_tokens=300,
                    temperature=0.6,
                    top_p=0.9,
                    stop=["[/INST]"]
                )
                intermediate_summary = response['choices'][0]['text'].strip()
                new_intermediate.append(intermediate_summary)
            intermediate = new_intermediate

        return intermediate[0] if intermediate else ""

    def load_and_prepare_logs(self, character_name: str, date: str) -> pd.DataFrame:
        """Helper function to load and prepare logs with correct format."""
        try:
            logs_df = self.summarizer_utils.load_character_logs_by_date(character_name, date)
            if not isinstance(logs_df, pd.DataFrame) or logs_df.empty:
                return pd.DataFrame()
            
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
