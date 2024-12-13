# Installation steps

python -m venv .venv 
.venv/Scripts/activate

"..."\llm-eval-module.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

pip install -e .

might only work with:
"openai==1.57.2",
"ragas==0.2.8",
"langchain==0.3.11",