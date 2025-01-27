from config_loader import ConfigLoader

from llm_invoker import *

if __name__ == "__main__":
    query = "What is the main idea of the paper?"
    config = ConfigLoader.load_config()
    
    invoker = LLMInvoker(query, config)
    res = invoker.invoke()
    print(res["result"])