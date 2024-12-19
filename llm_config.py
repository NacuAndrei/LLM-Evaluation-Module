from enum import Enum
from typing import Dict

from langchain_core.language_models import BaseLanguageModel

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

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
   
    provider = config.get("provider") 

    if provider == LLMProvider.OPENAI.value:
        return ChatOpenAI(model=config['model'], max_tokens=config['max_tokens'], temperature=config['temperature'])
    
    if provider == LLMProvider.OLLAMA.value:
        return ChatOllama(model=config['model'], max_tokens=config['max_tokens'], temperature=config['temperature'])
    
    raise ValueError(f"Invalid provider: {provider}")
    
def get_embeddings_config(config: Dict) -> object:
    provider = config[ConfigProps.PROVIDER.value]

    if provider == LLMProvider.OPENAI.value:
        return OpenAIEmbeddings(model=config[ConfigProps.MODEL.value])
    
    if provider == LLMProvider.OLLAMA.value:
        return OllamaEmbeddings(model=config[ConfigProps.MODEL.value])

    raise ValueError(f"Invalid provider: {provider}")
    
    