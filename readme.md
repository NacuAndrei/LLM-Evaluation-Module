# Installation steps

python -m venv .venv 
.venv/Scripts/activate

"..."\llm-eval-module.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

pip install -e .

might only work with:
"openai==1.57.2",
"ragas==0.2.8",
"langchain==0.3.11",

Steps to use Ollama (local models - Llama 3.3, Phi 3, Mistral, Gemma 2 etc.):
- Download Ollama
- To use llama 3, type in terminal: ollama run llama3 - downloads llama3 locally
- You may ask questions in terminal to verify the download, then close the terminal
- pip install -e .