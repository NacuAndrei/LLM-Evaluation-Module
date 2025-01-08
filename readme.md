# Installation steps
Make sure you have at least Python version 3.12.4
## Create Virtual Environment
python -m venv .venv
.venv/Scripts/activate

\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

pip install -e .

Steps to use Ollama (local models - Llama 3.3, Phi 3, Mistral, Gemma 2 etc.):
- Download and install Ollama from https://ollama.com/download
- Restart VS Code
- Download llama3 locally by typing the below command in the terminal:
ollama run llama3
- You can verify it by asking questions in the terminal and Ctrl+D to exit
