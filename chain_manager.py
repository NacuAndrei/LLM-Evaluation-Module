from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm_provider import get_llm

class ChainManager:
    def __init__(self, config: dict, embeddings_llm):
        self.config = config
        self.embeddings_llm = embeddings_llm

    def setup_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        #vectorstore = FAISS.from_documents(documents=[], embedding=self.embeddings_llm)

        llm_to_be_evaluated = get_llm(self.config["llm_to_be_evaluated"])

        return (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm_to_be_evaluated
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
