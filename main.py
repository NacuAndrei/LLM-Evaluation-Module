from ingestion import *
from chain import *
from llm_invoker import *

if __name__ == "__main__":
    docs_path = "C:/Users/andnacu/Documents/llm-eval-module/data/react_data/2210.03629v3.pdf"
    model = "text-embedding-3-small"
    chunk_size = 500
    chunk_overlap = 30
    query = "What is the main idea of the paper?"
    invoker = LLMInvoker(docs_path, model, chunk_size, chunk_overlap, query)
    res = invoker.invoke()
    print(res["result"])