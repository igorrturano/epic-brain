#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate  # For Linux/Mac
# source .venv/Scripts/activate  # For Windows

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install other requirements
pip install -r requirements.txt

echo "Installation complete with CUDA support for llama-cpp-python" 