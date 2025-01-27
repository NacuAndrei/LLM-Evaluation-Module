from chain_manager import ChainManager

class LLMInvoker:
    def __init__(self, query, config):
        self.query = query
        self.chain_manager = ChainManager(config['llm_to_be_evaluated'], config['embedding'], config['vectorstore'])
        self.qa_chain = self.chain_manager.create_chain()

    def invoke(self):
        result = self.qa_chain.invoke(input={"input": self.query})
        new_result = {
            "query": result["input"],
            "result": result["answer"],
            "source_documents": result["context"],
        }
        return new_result