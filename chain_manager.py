import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from ingestion import DocumentIngestor
from langchain.hub import pull

from llm_provider import get_llm

class ChainManager:
    def __init__(self, llm_config, embedding_config, vectorstore_config):
        self.llm_config = llm_config
        self.docs_path = os.environ.get("DATASET_FILENAME")
        self.ingestor = DocumentIngestor(embedding_config, vectorstore_config)
        self.vectorstore = self.ingestor.ingest_docs(self.docs_path)

    def create_chain(self):
        retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(get_llm(self.llm_config, type="llm"), retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(self.vectorstore.as_retriever(), combine_docs_chain)
        return retrieval_chain