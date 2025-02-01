import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_cpp import Llama
from pandas import DataFrame

load_dotenv()

MAX_CONTEXT = os.getenv('MAX_CONTEXT', 1000)
MAX_THREADS = os.getenv('MAX_THREADS', 4)
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


def remove_duplicates(texts):
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts


def load_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    log_files = find_logs_by_character(character_name, logs_dir)
    all_texts = []

    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as file:
            all_texts.extend(extract_text_from_log(file))

    unique_texts = remove_duplicates(all_texts)

    return "\n".join(unique_texts).strip()


def load_character_logs_by_date(character_name: str, date: str, logs_dir: str = LOGS_DIR) -> str:
    date_path = os.path.join(logs_dir, date)
    if not os.path.isdir(date_path):
        return ""

    log_path = os.path.join(date_path, f"{character_name}.log")
    if not os.path.exists(log_path):
        return ""

    with open(log_path, "r", encoding="utf-8") as file:
        texts = extract_text_from_log(file)

    unique_texts = remove_duplicates(texts)

    return "\n".join(unique_texts).strip()

def find_logs_by_character(character_name: str, logs_dir: str = LOGS_DIR) -> DataFrame:
    if not os.path.isdir(logs_dir):
        return pd.DataFrame()

    log_files = []
    for date_folder in os.listdir(logs_dir):
        date_path = os.path.join(logs_dir, date_folder)
        if os.path.isdir(date_path):
            log_path = os.path.join(date_path, f"{character_name}.log")
            if os.path.exists(log_path):
                log_files.append(log_path)

    return pd.DataFrame(log_files, columns=["data", "personagem", "raça", "coordenadas", "acao_fala", "outros"])