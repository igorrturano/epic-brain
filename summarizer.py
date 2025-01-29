import os
import csv
from typing import List
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

LOGS_DIR = os.getenv('LOG_PATH', '.')
MODEL_PATH = os.getenv('MODEL_PATH', '')


def get_llm(n_ctx: int = 2048, n_threads: int = 4) -> Llama:
    return Llama(model_path=MODEL_PATH, n_ctx=n_ctx, n_threads=n_threads)


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_logs(logs: str, max_tokens: int = 2048, chunk_size: int = 1000) -> str:
    logs = logs.strip()
    if not logs:
        return "Nenhum log encontrado para sumarização."

    llm = get_llm()
    chunks = chunk_text(logs, chunk_size)
    partial_summaries = []

    for chunk in chunks:
        prompt = f"Resuma as ações e falas do personagem nos seguintes logs:\n{chunk}"
        response = llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
        partial_summaries.append(response['choices'][0]['text'].strip())

    combined_summary = "\n".join(partial_summaries)
    final_prompt = f"Resuma as ações e falas do personagem com base nos resumos a seguir:\n{combined_summary}"
    final_response = llm(final_prompt, max_tokens=max_tokens, stop=["\n\n"])
    return final_response['choices'][0]['text'].strip()


def find_logs_by_character(character_name: str, logs_dir: str = LOGS_DIR) -> List[str]:
    if not os.path.isdir(logs_dir):
        return []

    log_files = []
    for date_folder in os.listdir(logs_dir):
        date_path = os.path.join(logs_dir, date_folder)
        if os.path.isdir(date_path):
            log_path = os.path.join(date_path, f"{character_name}.log")
            if os.path.exists(log_path):
                log_files.append(log_path)

    return log_files


def extract_text_from_log(file) -> List[str]:
    reader = csv.reader(file)
    return [
        row[4].strip()
        for row in reader
        if len(row) > 4 and row[4].strip() and row[4].strip().lower() != "none"
    ]


def load_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    log_files = find_logs_by_character(character_name, logs_dir)
    all_texts = []

    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as file:
            all_texts.extend(extract_text_from_log(file))

    return "\n".join(all_texts).strip()


def load_character_logs_by_date(character_name: str, date: str, logs_dir: str = LOGS_DIR) -> str:
    date_path = os.path.join(logs_dir, date)
    if not os.path.isdir(date_path):
        return ""

    log_path = os.path.join(date_path, f"{character_name}.log")
    if not os.path.exists(log_path):
        return ""

    with open(log_path, "r", encoding="utf-8") as file:
        texts = extract_text_from_log(file)
    return "\n".join(texts).strip()


def load_and_summarize_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    logs = load_character_logs(character_name, logs_dir)
    if not logs:
        return f"Nenhum log encontrado para o personagem {character_name}."

    return summarize_logs(logs)

def load_and_summarize_character_logs_by_date(character_name: str, date: str, logs_dir: str = LOGS_DIR) -> str:
    logs = load_character_logs_by_date(character_name, date, logs_dir)
    if not logs:
        return f"Nenhum log encontrado para o personagem {character_name}."

    return summarize_logs(logs)
