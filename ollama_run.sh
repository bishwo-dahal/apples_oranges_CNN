#!/bin/bash

source /opt/miniforge/bin/activate 
# start ollama after extracting

/bin/ollama serve&

# download astrollama
# /bin/ollama pull hf.co/UniverseTBD/astrollama.gguf:latest
echo "Completed installing astrollama"
python3 astrollama_benchmark.py

