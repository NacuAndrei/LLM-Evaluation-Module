from chain_manager import ChainManager

class LLMInvoker:
    def __init__(self, questions, config):
        self.questions = questions
        self.chain_manager = ChainManager(config['llm_to_be_evaluated'], config['embedding'], config['vectorstore'])
        self.chain = self.chain_manager.create_chain()
        self.vectorstore = self.chain_manager.vectorstore

    def invoke(self):
        results = []
        for question_data in self.questions:
            question = question_data["question"]
            ground_truth = question_data["ground_truth"]
            
            result = self.chain.invoke(input={"input": question})
            context_docs = self.chain_manager.ingestor.similarity_search(self.vectorstore, question)
            context = " ".join([doc.page_content for doc in context_docs])
            
            new_result = {
                "question": result["input"],
                "answer": result["answer"],
                "ground_truth": ground_truth,
                "context": context
            }
            results.append(new_result)
        return results