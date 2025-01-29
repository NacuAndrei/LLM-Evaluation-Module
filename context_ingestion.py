from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from llm_provider import get_llm

class DocumentIngestor:
    def __init__(self, embedding_config, vectorstore_config):
        self.embedding = get_llm(embedding_config, type="embeddings")
        self.chunk_size = vectorstore_config["chunk_size"]
        self.chunk_overlap = vectorstore_config["chunk_overlap"]

    def ingest_docs(self, docs_path):
        loader = PyPDFLoader(docs_path)
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splitted_documents = text_splitter.split_documents(raw_documents)
        vectorstore = FAISS.from_documents(documents=splitted_documents, embedding=self.embedding)
        return vectorstore

    def similarity_search(self, vectorstore, query, k=5):
        return vectorstore.similarity_search(query, k=k)