import os

from dotenv import load_dotenv

from summarizer_utils import get_llm, chunk_text, load_character_logs, \
    load_character_logs_by_date, LOGS_DIR

load_dotenv()

SUMMARY_PROMPT = os.getenv('SUMMARY_PROMPT', '')


def summarize_large_logs(character_name: str, logs: str, max_tokens: int = 5000) -> Exception | str:
    logs = logs.strip()
    if not logs:
        return "Nenhum log encontrado para sumarização."

    try:
        llm = get_llm()
        chunks = chunk_text(logs, chunk_size=max_tokens)
        partial_summaries = []

        for chunk in chunks:
            prompt = SUMMARY_PROMPT.format(character_name=character_name, chunk=chunk)
            response = llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
            partial_summaries.append(response['choices'][0]['text'].strip())

        combined_summary = "\n".join(logs)
        final_prompt = SUMMARY_PROMPT.format(character_name=character_name, chunk=combined_summary)
        final_response = llm(final_prompt, max_tokens=max_tokens, stop=["\n\n"])
        return final_response['choices'][0]['text'].strip()
    except Exception as e:
        return e


def summarize_logs(character_name: str, logs: str, max_tokens: int = 1000) -> Exception | str:
    logs = logs.strip()
    if not logs:
        return "Nenhum log encontrado para sumarização."

    try:
        llm = get_llm()
        final_prompt = SUMMARY_PROMPT.format(character_name=character_name, chunk=logs)
        final_response = llm(final_prompt, max_tokens=max_tokens, stop=["\n\n"])
        return final_response['choices'][0]['text'].strip()
    except Exception as e:
        return e


def load_and_summarize_character_logs(character_name: str, logs_dir: str = LOGS_DIR) -> str:
    logs = load_character_logs(character_name, logs_dir)
    if not logs:
        return f"Nenhum log encontrado para o personagem {character_name}."

    return summarize_logs(character_name, logs)


def load_and_summarize_character_logs_by_date(character_name: str, date: str, logs_dir: str = LOGS_DIR) -> str:
    logs = load_character_logs_by_date(character_name, date, logs_dir)
    if not logs:
        return f"Nenhum log encontrado para o personagem {character_name}."

    return summarize_logs(character_name, logs)
