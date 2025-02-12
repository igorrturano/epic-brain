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
                                max_prompt_tokens: int = 800) -> str:
        """
        Build a prompt dynamically by adding one interaction (log) at a time,
        ensuring the total prompt (instructions + logs) stays within a token limit.
        """
        try:
            print(f"  - Processando chunk de interações entre {main_character} e {other_character}")
            
            # Ensure each dataframe has a datetime column.
            for df in [main_logs, char_logs]:
                if 'datetime' not in df.columns:
                    df['datetime'] = pd.to_datetime(
                        df['date'],
                        format='%d-%m-%Y %H:%M:%S',
                        errors='coerce'
                    )
            
            # Concatenate and sort logs by datetime.
            combined_logs = pd.concat([main_logs, char_logs])
            combined_logs = combined_logs.sort_values('datetime', na_position='last')
            
            # Define the fixed instruction parts.
            instruction_prefix = f"[INST] Analise e resuma em português as interações entre {main_character} e {other_character}:\n\n"
            instruction_suffix = ("\n\nForneça um breve resumo focando em:\n"
                                  "1. O tipo de interação entre eles\n"
                                  "2. O tom da conversa\n"
                                  "3. Ações importantes realizadas\n"
                                  "4. Locais onde interagiram (baseado nas coordenadas)\n\n"
                                  "Resumo: [/INST]")
            
            log_lines = []
            # Add one log (formatted as a line) at a time until the prompt is near the limit.
            for _, log in combined_logs.iterrows():
                if pd.notna(log['date']) and pd.notna(log['say']):
                    coord = log.get('coord', 'Unknown location')
                    char_name = log.get('character', 'Unknown')
                    log_line = f"[{log['date']}] ({coord}) {char_name}: {log['say']}"
                    # Build a candidate prompt including the new line.
                    candidate_text = instruction_prefix + "\n".join(log_lines + [log_line]) + instruction_suffix
                    if approximate_token_count(candidate_text) > max_prompt_tokens:
                        # Reached near the token limit; stop adding new lines.
                        break
                    log_lines.append(log_line)
            
            if not log_lines:
                print("    - Nenhuma interação cabe no prompt limitado.")
                return ""
            
            logs_text = "\n".join(log_lines)
            prompt = instruction_prefix + logs_text + instruction_suffix
            # (Optional) Uncomment the following line to see the approximate token count.
            # print(f"Prompt token count: {approximate_token_count(prompt)}")
            
            response = self.summarizer_utils.llm(
                prompt,
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
            print(traceback.format_exc())
            return ""

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
            
            # Get and prepare main character logs.
            main_logs = self.load_and_prepare_logs(main_character, date)
            if main_logs.empty:
                return f"Erro: Não foi possível carregar os logs de {main_character}"
            
            valid_summaries = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for char in nearby_chars:
                    print(f"\nAnalisando interações com {char}...")
                    char_logs = self.load_and_prepare_logs(char, date)
                    
                    if char_logs.empty:
                        print(f"  - Pulando {char} - logs não encontrados")
                        continue
                    
                    # Here we still split the logs into chunks (by row count) for parallel processing.
                    for i in range(0, len(char_logs), chunk_size):
                        char_chunk = char_logs.iloc[i:i + chunk_size].copy()
                        # For the main character, we take a chunk from the beginning (or adjust as needed).
                        main_chunk = main_logs.iloc[0:min(chunk_size, len(main_logs))].copy()
                        # Use the new dynamic chunk builder by specifying max_prompt_tokens.
                        future = executor.submit(
                            self.process_character_chunk,
                            main_character,
                            char,
                            main_chunk,
                            char_chunk,
                            800  # max_prompt_tokens
                        )
                        futures.append((char, future))
                
                for char, future in futures:
                    result = future.result()
                    if result and len(result.strip()) > 0:
                        print(f"  - Resumo válido gerado para interação com {char}")
                        valid_summaries.append(f"Interações com {char}:\n{result}")
            
            if not valid_summaries:
                return f"Não foi possível gerar resumos válidos para as interações de {main_character}"
            
            print(f"\nGerando resumo final de {len(valid_summaries)} interações válidas...")
            
            final_result = self.hierarchical_summary(valid_summaries, main_character)
            
            print("\nResumo final gerado com sucesso!")
            return final_result
            
        except Exception as e:
            print(f"\nErro em summarize_character_interactions: {str(e)}")
            return str(e)
    
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
