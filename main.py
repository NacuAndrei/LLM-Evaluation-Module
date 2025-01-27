from config_loader import ConfigLoader

from llm_invoker import *

if __name__ == "__main__":
    questions = [
        "What is the main idea of the paper?",
        "How does the proposed method work?",
        "What is the most important key finding of the study?"
    ]
    config = ConfigLoader.load_config()
    
    invoker = LLMInvoker(questions, config)
    results = invoker.invoke()
    
    for res in results:
        print(f"Question: {res['query']}")
        print(f"Answer: {res['result']}\n")