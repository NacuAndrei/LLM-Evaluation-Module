from enum import Enum
from typing import Dict

from langchain_core.language_models import BaseLanguageModel

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

class LLM_Names(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    
class LLM_IDs(Enum):
    OPENAI_GPT_4o_MINI = "gpt-4o-mini"
    OLLAMA_LLAMA_3 = "llama3"
  
class ConfigProps(Enum):
    PROVIDER = "provider"
    MODEL_ID = "model_id"
    MODEL = "model"
  
def get_llm(config: Dict, type = "llm") -> BaseLanguageModel:
    relevant_llm_keys = {'model', 'max_tokens', 'temperature'}
    valid_config = {k: v for k, v in config.items() if k in relevant_llm_keys}
    
    provider = config.get(ConfigProps.PROVIDER.value)
    
    if type == "llm":
        if provider == LLM_Names.OPENAI.value:
            return ChatOpenAI(**valid_config)
        
        if provider == LLM_Names.OLLAMA.value:
            return ChatOllama(**valid_config)
    
        raise ValueError(f"Invalid llm: {provider}")

    elif type == "embeddings":
        if provider == LLM_Names.OPENAI.value:
            return OpenAIEmbeddings(**valid_config)
    
        if provider == LLM_Names.OLLAMA.value:
            return OllamaEmbeddings(**valid_config)

    else:
        raise ValueError(f"Invalid provider: {provider}")

    