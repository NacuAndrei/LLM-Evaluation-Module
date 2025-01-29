from enum import Enum
from typing import Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

class LLM_Names(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
  
class ConfigProps(Enum):
    PROVIDER = "provider"
    MODEL = "model"
  
def get_llm(config: Dict, type="llm"):
    provider = config.get(ConfigProps.PROVIDER.value)
    
    if provider not in LLM_Names._value2member_map_:
        raise ValueError(f"Invalid provider: {provider}")
    
    config_filtered = {k: v for k, v in config.items() if k != ConfigProps.PROVIDER.value}
    
    if type == "llm":
        llm_classes = {
            LLM_Names.OPENAI.value: ChatOpenAI,
            LLM_Names.OLLAMA.value: ChatOllama
        }
    elif type == "embeddings":
        llm_classes = {
            LLM_Names.OPENAI.value: OpenAIEmbeddings,
            LLM_Names.OLLAMA.value: OllamaEmbeddings
        }
    else:
        raise ValueError(f"Invalid type: {type}")
    
    return llm_classes[provider](**config_filtered)