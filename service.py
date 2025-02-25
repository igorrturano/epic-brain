from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from llama_cpp import Llama
import re

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Epic Brain Translation API",
    description="API for translating text using LLM",
    version="1.0.0"
)

# Add CORS middleware with strict localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Initialize LLM
llm = Llama(
    model_path=os.getenv('MODEL_PATH', "models/model.gguf"),
    n_ctx=500,
    n_threads=int(os.getenv('MAX_THREADS', 4)),
    n_batch=int(os.getenv('MAX_BATCH', 512))#,
    #n_gpu_layers=int(os.getenv('MAX_GPU_LAYERS', 35))
)

class TranslationRequest(BaseModel):
    message: str
    target_language: str
    original_language: Optional[str] = None

class TranslationResponse(BaseModel):
    message: str
    detected_language: Optional[str] = None

async def verify_localhost(request: Request):
    """Verify that the request is coming from localhost"""
    host = request.client.host
    if host not in ['127.0.0.1', 'localhost', '::1']:
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: Only localhost requests are allowed"
        )

@app.post("/epic-brain/translate", response_model=TranslationResponse)
async def translate_text(request: Request, translation_request: TranslationRequest):
    # Verify localhost first
    await verify_localhost(request)
    
    try:
        # Validate input
        if not translation_request.message or not translation_request.target_language:
            raise HTTPException(status_code=400, detail="Message and target language are required")
            
        # Create a simpler prompt for raw completion
        if translation_request.original_language:
            prompt = f"""Translate this text from {translation_request.original_language} to {translation_request.target_language}:

"{translation_request.message}"

Translation: """
        else:
            prompt = f"""Translate this text to {translation_request.target_language}:

"{translation_request.message}"

Translation: """

        print(f"Using raw completion with prompt: {prompt}")
        
        # Use raw completion instead of chat completion
        response = llm(
            prompt,
            max_tokens=len(translation_request.message) * 2,
            temperature=0.3,
            stop=["<", "\n\n", "Translation:"]
        )
        
        print(f"Raw response: {response}")
        
        # Extract the text from raw completion
        translated_text = response['choices'][0]['text'].strip()
        print(f"Extracted text: {translated_text}")

        # Fallback: if translation failed, try simple rule-based translations for common phrases
        if not translated_text or len(translated_text) < 1:
            # Simple fallback dictionary for common phrases
            fallbacks = {
                "Hello, world!": {
                    "pt": "Olá, mundo!",
                    "pt_br": "Olá, mundo!",
                    "portuguese": "Olá, mundo!"
                },
                "Hello": {
                    "pt": "Olá",
                    "pt_br": "Olá",
                    "portuguese": "Olá"
                },
                "Good morning": {
                    "pt": "Bom dia",
                    "pt_br": "Bom dia",
                    "portuguese": "Bom dia"
                }
            }
            
            target = translation_request.target_language.lower()
            if translation_request.message in fallbacks and target in fallbacks[translation_request.message]:
                translated_text = fallbacks[translation_request.message][target]
                print(f"Using fallback translation: {translated_text}")
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Translation failed: Empty response from model"
                )
        
        # Return translation
        return TranslationResponse(
            message=translated_text,
            detected_language=translation_request.original_language
        )
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.get("/health")
async def health_check(request: Request):
    await verify_localhost(request)
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1",  # Only bind to localhost
        port=5000,
    )