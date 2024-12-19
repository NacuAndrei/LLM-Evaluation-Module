from enum import Enum
from typing import Dict

from langchain_core.language_models import BaseLanguageModel

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    
class LLMID(Enum):
    OPENAI_GPT_4o_MINI = "gpt-4o-mini"
    OLLAMA_LLAMA_3 = "llama3"
  
class ConfigProps(Enum):
    PROVIDER = "provider"
    MODEL_ID = "model_id"
    MODEL = "model"
  
def get_llm_config(config: Dict) -> BaseLanguageModel:
    relevant_llm_keys = {'model', 'max_tokens', 'temperature'}
    valid_config = {k: v for k, v in config.items() if k in relevant_llm_keys}
    
    provider = config[ConfigProps.PROVIDER.value]
    
    if provider == LLMProvider.OPENAI.value:
        return ChatOpenAI(**valid_config)
    
    if provider == LLMProvider.OLLAMA.value:
        return ChatOllama(**valid_config)
    
    raise ValueError(f"Invalid provider: {provider}")
    
    
    