import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_cpp import Llama
from pandas import DataFrame

load_dotenv()

MAX_CONTEXT = int(os.getenv('MAX_CONTEXT', 1000))
MAX_THREADS = int(os.getenv('MAX_THREADS', 4))
MODEL_PATH = os.getenv('MODEL_PATH', "models/model.gguf")
LOGS_DIR = os.getenv('LOG_PATH', '.')


def calculate_distance(coord1, coord2):
    x1, y1, z1 = map(int, coord1.split(':'))
    x2, y2, z2 = map(int, coord2.split(':'))
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def get_llm(n_ctx: int = MAX_CONTEXT, n_threads: int = MAX_THREADS) -> Llama:
    return Llama(model_path=MODEL_PATH, n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def load_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    log_files = find_logs_by_character(character_name, logs_dir)

    unique_texts = log_files.drop_duplicates(subset=['say'], keep='first')

    return "\n".join(unique_texts['say'].values)


def load_character_logs_by_date(character_name: str, date: str, logs_dir: str = LOGS_DIR) -> str:
    date_path = os.path.join(logs_dir, date)
    if not os.path.isdir(date_path):
        return ""

    log_path = os.path.join(date_path, f"{character_name}.log")
    if not os.path.exists(log_path):
        return ""

    log_files = find_logs_by_character(character_name, log_path)
    unique_texts = log_files.drop_duplicates(subset=['say'], keep='first')

    return "\n".join(unique_texts['say'].values)

def find_logs_by_character(character_name: str, logs_dir: str = LOGS_DIR) -> DataFrame:
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