import os
from typing import List
from dotenv import load_dotenv
from llama_cpp import Llama

# Caminho para o modelo GGML
MODEL_PATH = "models/mistral-7b-v0.1.Q4_K_M.gguf"  # Substitua pelo caminho do seu modelo

# Carrega o modelo
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)  # Ajuste o número de threads conforme a VPS

load_dotenv()
LOGS_DIR = os.getenv('LOG_PATH')

def summarize_logs(logs: str, max_tokens: int = 200) -> str:
    if not logs:
        return "Nenhum log encontrado para sumarização."

    prompt = f"Resuma as ações e falas do personagem nos seguintes logs:\n{logs}"

    response = llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
    return response['choices'][0]['text'].strip()

def load_and_summarize_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    logs = load_character_logs(character_name, logs_dir)
    if not logs:
        return f"Nenhum log encontrado para o personagem {character_name}."

    summary = summarize_logs(logs)
    return summary

def list_characters(logs_dir: str = LOGS_DIR) -> List[str]:
    characters = set()
    for date_folder in os.listdir(logs_dir):
        date_path = os.path.join(logs_dir, date_folder)
        if os.path.isdir(date_path):
            for log_file in os.listdir(date_path):
                if log_file.endswith(".log"):
                    character_name = log_file.replace(".log", "")
                    characters.add(character_name)
    return sorted(list(characters))

def find_logs_by_character(character_name: str, logs_dir: str = LOGS_DIR) -> List[str]:
    log_files = []
    for date_folder in os.listdir(logs_dir):
        date_path = os.path.join(logs_dir, date_folder)
        if os.path.isdir(date_path):
            log_path = os.path.join(date_path, f"{character_name}.log")
            if os.path.exists(log_path):
                log_files.append(log_path)
    return log_files

def load_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    log_files = find_logs_by_character(character_name, logs_dir)
    all_logs = ""
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as file:
            all_logs += file.read() + "\n"
    return all_logs.strip()

def load_character_logs_by_date(character_name: str, date: str, logs_dir: str = LOGS_DIR) -> str:
    date_path = os.path.join(logs_dir, date)
    if not os.path.exists(date_path):
        return ""

    log_path = os.path.join(date_path, f"{character_name}.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""