from chain import Chain

class LLMInvoker:
    def __init__(self, docs_path, model, chunk_size, chunk_overlap, query):
        self.chain = Chain(docs_path, model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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