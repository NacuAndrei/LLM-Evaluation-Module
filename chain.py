from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI
from ingestion import DocumentIngestor
from langchain.hub import pull

class Chain:
    def __init__(self, docs_path, model="text-embedding-3-small", chunk_size=500, chunk_overlap=30):
        self.docs_path = docs_path
        self.ingestor = DocumentIngestor(model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vectorstore = self.ingestor.ingest_docs(docs_path)

    def create_chain(self):
        retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(self.vectorstore.as_retriever(), combine_docs_chain)
        
        return retrieval_chain