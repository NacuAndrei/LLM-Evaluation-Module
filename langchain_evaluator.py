import pandas as pd

from langchain.evaluation import load_evaluator

from llm_invoker import *

class LangchainEvaluator:
    def __init__(self, questions, config):
        self.evaluator = load_evaluator("labeled_criteria", criteria="correctness")
        self.questions = questions
        self.config = config
    
    def evaluate(self):
        invoker = LLMInvoker(self.questions, self.config)
        results = invoker.invoke()
        
        eval_results = []
        for res in results:
            eval_result = self.evaluator.evaluate_strings(
            input=res["question"],
            prediction=res["answer"],
            reference=res["ground_truth"]
            )
            eval_results.append({
            "question": res["question"],
            "answer": res["answer"],
            "ground_truth": res["ground_truth"],
            "reasoning": eval_result["reasoning"],
            "score": eval_result["score"]
            })
        
        df = pd.DataFrame(eval_results)
        return df