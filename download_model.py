import os
import requests
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)
MODEL_URL = os.getenv('MODEL_URL')
MODELS_DIR = os.getenv('MODELS_DIR')
MODEL_PATH = os.path.join(MODELS_DIR, "model.gguf")

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        print(f"Downloading models from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Model saved {MODEL_PATH}")
        else:
            print(f"Error to save model: {response.status_code}")
    else:
        print(f"Model already downloaded: {MODEL_PATH}")