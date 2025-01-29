from config_loader import ConfigLoader
from langchain_evaluator import LangchainEvaluator
from llm_invoker import *
from excel_writer import *
import json

if __name__ == "__main__":
    config = ConfigLoader.load_config()
    
    with open('examples/react_paper.json', 'r') as file:
        questions = json.load(file)
        
    evaluator = LangchainEvaluator(questions, config)
    df = evaluator.evaluate()
    
    writer = ExcelWriter(config=config)
    writer.write_dataframe(df, sheet_name="Langchain_Evaluation")
    
    print("Done!")
    