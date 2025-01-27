from chain_manager import ChainManager

class LLMInvoker:
    def __init__(self, questions, config):
        self.questions = questions
        self.chain_manager = ChainManager(config['llm_to_be_evaluated'], config['embedding'], config['vectorstore'])
        self.chain = self.chain_manager.create_chain()

    def invoke(self):
        results = []
        for question in self.questions:
            result = self.chain.invoke(input={"input": question})
            new_result = {
                "query": result["input"],
                "result": result["answer"],
                #"source_documents": result["context"],
            }
            results.append(new_result)
        return results