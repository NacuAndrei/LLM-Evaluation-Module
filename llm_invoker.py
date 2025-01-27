import os
from chain import Chain

class LLMInvoker:
    def __init__(self, query, config: dict):
        self.config = config
        
        self.chain = Chain(self.config)
        self.qa_chain = self.chain.create_chain()
        
        self.query = query

    def invoke(self):
        result = self.qa_chain.invoke(input={"input": self.query})
        new_result = {
            "query": result["input"],
            "result": result["answer"],
            "source_documents": result["context"],
        }
        
        return new_result