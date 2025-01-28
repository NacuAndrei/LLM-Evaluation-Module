from config_loader import ConfigLoader
from langchain_evaluator import LangchainEvaluator
from llm_invoker import *
import json

if __name__ == "__main__":
    config = ConfigLoader.load_config()
    
    with open('examples/react_paper.json', 'r') as file:
        questions = json.load(file)
        
    evaluator = LangchainEvaluator(questions, config)
    
    df = evaluator.evaluate()
    print(df)