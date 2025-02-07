import os

from dotenv import load_dotenv

from summarizer_utils import get_llm, load_character_logs, \
    load_character_logs_by_date, LOGS_DIR

load_dotenv()

SUMMARY_PROMPT = os.getenv('SUMMARY_PROMPT', '')

def create_chat_template(prompt: str) -> str:
    return f"""<｜User｜> {prompt} <｜Assistant｜>"""


def summarize_logs(character_name: str, logs: str, max_tokens: int = 1000) -> Exception | str:
    logs = logs.strip()
    if not logs:
        return "Nenhum log encontrado para sumarização."

    try:
        llm = get_llm()
        prompt = SUMMARY_PROMPT.format(
            character_name=character_name,
            logs=logs
        )
        formatted_prompt = create_chat_template(prompt)

        response = llm(formatted_prompt, 
                      max_tokens=max_tokens,
                      temperature=0.7,
                      top_p=0.9,
                      stop=["<|Assistant|>", "\n\n"])
        

        return response['choices'][0]['text'].strip()
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
        return f"Nenhum log encontrado para o personagem {character_name} na data {date}."

    return summarize_logs(character_name, logs)
