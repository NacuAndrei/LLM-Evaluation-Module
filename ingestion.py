from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class DocumentIngestor:
    #TODO: Save the vectorstore to a file
    def __init__(self, config):
        self.config = config
        
        self.embedding_config = self.config["embedding"]
        self.vectorstore_config = self.config["vectorstore"]
        
        self.embedding = OpenAIEmbeddings(model=self.embedding_config["model"])
        self.chunk_size = self.vectorstore_config["chunk_size"]
        self.chunk_overlap = self.vectorstore_config["chunk_overlap"]

    def ingest_docs(self, docs_path):
        loader = PyPDFLoader(docs_path)

        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splitted_documents = text_splitter.split_documents(raw_documents)
        
        vectorstore = FAISS.from_documents(documents=splitted_documents, embedding=self.embedding)
        
        return vectorstore