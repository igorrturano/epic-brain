import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_cpp import Llama
from pandas import DataFrame
from dataclasses import dataclass

@dataclass
class Config:
    max_context: int
    max_threads: int
    max_batch: int
    max_gpu_layers: int
    model_path: str
    logs_dir: str

class SummarizerUtils:
    def __init__(self):
        load_dotenv(verbose=True, override=True)
        self.config = self._initialize_config()
        self.llm = self._initialize_llm()
        
    def _initialize_config(self) -> Config:
        config = Config(
            max_context=int(os.getenv('MAX_CONTEXT', '8192')),
            max_threads=int(os.getenv('MAX_THREADS', '8')),
            max_batch=int(os.getenv('MAX_BATCH', '512')),
            max_gpu_layers=int(os.getenv('MAX_GPU_LAYERS', '35')),
            model_path=os.getenv('MODEL_PATH', "models/model.gguf"),
            logs_dir=os.getenv('LOG_PATH', '.')
        )
        print(f"Initialized config: Context={config.max_context}, "
              f"Threads={config.max_threads}, Batch={config.max_batch}, "
              f"GPU Layers={config.max_gpu_layers}")
        return config
    
    def _initialize_llm(self) -> Llama:
        return Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.max_context,
            n_threads=self.config.max_threads,
            n_batch=self.config.max_batch,
            #n_gpu_layers=self.config.max_gpu_layers,
            verbose=False
        )
    
    def find_logs_by_character(self, character_name: str, logs_dir: str = None) -> DataFrame:
        logs_dir = logs_dir or self.config.logs_dir
        log_files = []
        
        # If logs_dir is a specific log file
        if os.path.isfile(logs_dir):
            with open(logs_dir, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    try:
                        # Split the line into components
                        parts = line.strip().split(',')
                        if len(parts) >= 5:  # Ensure we have all required fields
                            log_files.append({
                                'date': parts[0],
                                'character': parts[1],
                                'race': parts[2],
                                'coord': parts[3],
                                'say': parts[4],
                                'etc': parts[5] if len(parts) > 5 else ''
                            })
                    except Exception:
                        continue
        
            return pd.DataFrame(log_files)

        # If logs_dir is a directory containing log files
        for date_folder in os.listdir(logs_dir):
            date_path = os.path.join(logs_dir, date_folder)
            if os.path.isdir(date_path):
                log_path = os.path.join(date_path, f"{character_name}.log")
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line in lines:
                            try:
                                parts = line.strip().split(',')
                                if len(parts) >= 5:
                                    log_files.append({
                                        'date': parts[0],
                                        'character': parts[1],
                                        'race': parts[2],
                                        'coord': parts[3],
                                        'say': parts[4],
                                        'etc': parts[5] if len(parts) > 5 else ''
                                    })
                            except Exception:
                                continue

        return pd.DataFrame(log_files)
    
    def load_prompt_template() -> str:
        try:
            with open('summarization_prompt.txt', 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            print("Warning: summarization_prompt.txt not found, using default prompt")
            return """Você é um assistente que fala português. Analise os logs e resuma o comportamento do personagem {character_name}.\n\nLogs:\n{logs}\n\nResumo:"""


    def calculate_distance(self, coord1: str, coord2: str) -> float:
        try:
            if not coord1 or not coord2 or coord1 == '<uninitialized object>' or coord2 == '<uninitialized object>':
                return float('inf')
            
            x1, y1, z1 = map(int, str(coord1).split(':'))
            x2, y2, z2 = map(int, str(coord2).split(':'))
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        except (ValueError, TypeError, AttributeError):
            return float('inf')

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def load_character_logs(self, character_name: str, logs_dir: str = None) -> DataFrame:
        log_files = self.find_logs_by_character(character_name, logs_dir)

        unique_texts = log_files.drop_duplicates(subset=['say'], keep='first')

        return unique_texts
    
    def load_character_logs_by_date(self, character_name: str, date: str) -> str:
        date_path = os.path.join(self.config.logs_dir, date)
        if not os.path.isdir(date_path):
            return ""

        log_path = os.path.join(date_path, f"{character_name}.log")
        if not os.path.exists(log_path):
            return ""

        log_files = self.find_logs_by_character(character_name, log_path)
        unique_texts = log_files.drop_duplicates(subset=['say'], keep='first')

        return unique_texts

    def find_nearby_characters(self, main_character: str, date: str, max_distance: float = 10.0) -> List[str]:
        # Get main character logs for the date
        main_logs = self.load_character_logs_by_date(main_character, date)
        if main_logs.empty:
            return []
        
        date_path = os.path.join(self.config.logs_dir, date)
        nearby_characters = set()
        
        # Convert dates to datetime with explicit format
        main_logs['datetime'] = pd.to_datetime(
            main_logs['date'], 
            format='%d-%m-%Y %H:%M:%S', 
            dayfirst=True
        )
        
        for log_file in os.listdir(date_path):
            if not log_file.endswith('.log'):
                continue
            
            character_name = log_file.replace('.log', '')
            if character_name == main_character:
                continue
            
            character_logs = self.load_character_logs_by_date(character_name, date)
            if character_logs.empty:
                continue
            
            # Convert character logs dates
            character_logs['datetime'] = pd.to_datetime(
                character_logs['date'], 
                format='%d-%m-%Y %H:%M:%S', 
                dayfirst=True
            )
            
            # Compare coordinates for each time period
            for _, main_row in main_logs.iterrows():
                if pd.isna(main_row.get('coord')) or main_row.get('coord') == '<uninitialized object>':
                    continue
                
                main_time = main_row['datetime']
                
                # Find logs within the same time window (5 minutes)
                time_window = character_logs[
                    (character_logs['datetime'] - main_time).abs() <= pd.Timedelta(minutes=5)
                ]
                
                for _, char_row in time_window.iterrows():
                    if pd.isna(char_row.get('coord')) or char_row.get('coord') == '<uninitialized object>':
                        continue
                    
                    try:
                        distance = self.calculate_distance(main_row['coord'], char_row['coord'])
                        if distance <= max_distance:
                            nearby_characters.add(character_name)
                            break
                    except (ValueError, TypeError):
                        continue
        
        return list(nearby_characters)