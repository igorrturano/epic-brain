@echo off

REM Create and activate virtual environment if it doesn't exist
if not exist ".venv" (
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install llama-cpp-python with CUDA support
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python

REM Install other requirements
pip install -r requirements.txt

echo Installation complete with CUDA support for llama-cpp-python 