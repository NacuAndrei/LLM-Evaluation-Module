#General
import os
import sys
import json
import logging
import openai
from typing import List, Dict, Any
import unittest
import pandas as pd
from dotenv import load_dotenv
from ruamel.yaml import YAML
from datasets import Dataset
from datetime import datetime

#Ragas
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate

#Embeddings & VectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

#LLM To be Evaluated
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama


#Temporary
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

EVAL_DATASET_PATH = os.environ.get('DATASET_FILENAME')
print(EVAL_DATASET_PATH)
CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        embeddings = OllamaEmbeddings(model=self.config["embeddings"]["model"]) #OpenAIEmbeddings()
        self.embedding = LangchainEmbeddingsWrapper(embeddings)

        # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
        prompt = hub.pull("rlm/rag-prompt")

        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()

        # Split text in chunks and store it in FAISS vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

        #llm_to_be_evaluated
        self.llm_to_be_evaluated = ChatOllama( #ChatOpenAI(
            model=self.config["llm_to_be_evaluated"]["model"],
            temperature=self.config["llm_to_be_evaluated"]["temperature"]
        )        
        
        self.chain = (
            {
                "context": vectorstore.as_retriever() | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm_to_be_evaluated
            | StrOutputParser()
        )

        #Ragas LLM
        self.ragas_helper_llm = ChatOpenAI(
            model=self.config["ragas_helper_llm"]["model"],
            temperature=self.config["ragas_helper_llm"]["temperature"]
        )

        self.llm = LangchainLLMWrapper(self.ragas_helper_llm)
        self.init_ragas_metrics()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def init_ragas_metrics(self):
        self.metrics = []

        metric = FactualCorrectness()
        metric.llm = self.llm

        run_config = RunConfig()
        metric.init(run_config)

        self.metrics += [metric]
            
    def load_dataset(self, absolute_path: str):
        with open(absolute_path, "r") as json_file:
            all_samples = json.load(json_file)
        return all_samples
    
    def get_complete_sample(self, sample: Dict[str, Any]):
        question = sample["question"]
        prompt = f"Please formulate the answer for the following question in one or multiple sentences. This is the question: {question}"

        response = self.chain.invoke(prompt)
        
        sample["response"] = response

        return sample

    def get_evaluation_batch(self):
        evaluation_batch = {
            "question": [],
            "ground_truth": [],
            "response": []
        }
        
        self.incomplete_samples = self.load_dataset(absolute_path=EVAL_DATASET_PATH)

        for incomplete_sample in self.incomplete_samples:
            complete_sample = self.get_complete_sample(incomplete_sample)

            evaluation_batch["question"].append(complete_sample["question"])
            evaluation_batch["ground_truth"].append(complete_sample["ground_truth"])
            evaluation_batch["response"].append(complete_sample["response"])

        return evaluation_batch
    
    def write_results_to_excel(self, results):
        new_data = {
            "Version Number": [datetime.now().strftime("%Y-%m-%dT%H:%M")],
            "Doc Format": [self.config["test"]["doc_format"]],
            "Number of Questions": [len(self.incomplete_samples)],
            # "top_k": [self.chain_config["retriever"]["search_kwargs"]["k"]],
            # "chunk_size": [CHUNK_SIZES[self.chain_config["ingestion"]["chunk_size"]]],
            # "chunk_overlap": [self.chain_config["ingestion"]["overlap"]],
            "Embeddings Model": [self.config["embeddings"]["model"]],
            "Model to be Evaluated": [self.config["llm_to_be_evaluated"]["model"]],
            "Model used for Ragas Metrics": [self.config["ragas_helper_llm"]["model"]],
            "Questions": [sample["question"] for sample in self.incomplete_samples],
            "Answers": [sample["response"] for sample in self.incomplete_samples],
        }
        for metric in self.metrics:
            new_data[metric.name] = results[metric.name]
            
        max_length = max(len(lst) for lst in new_data.values())
    
        for key in new_data:
            current_length = len(new_data[key])
            if current_length < max_length:
                new_data[key].extend([None] * (max_length - current_length))
            
        df_new = pd.DataFrame(new_data)
        excel_path = os.environ["RESULTS_EXCEL_PATH"]
        
        if not os.path.exists(excel_path):
            df_new.to_excel(excel_path, index=False)
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                startrow = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
                df_new.to_excel(writer, index=False, header=writer.sheets['Sheet1'].max_row == 0, startrow=startrow)

    def run_experiment(self):
        evaluation_batch = self.get_evaluation_batch()
        ds = Dataset.from_dict(evaluation_batch)
        r = evaluate(ds, metrics=self.metrics)
        logger.info(f"Final scores {r}")
        self.write_results_to_excel(results=r)
     
class TestRagas(unittest.TestCase):

    yaml_config_file = YAML(typ="safe")
    with open(os.environ["CONFIG_PATH"], "r") as file:
        config = yaml_config_file.load(file)

    ragas_eval = RagasEvaluator(config=config)

    ragas_eval.run_experiment()


if __name__ == "__main__":
    # unittest.main(exit=False)
    dataset = RagasEvaluator(config=TestRagas.config).load_dataset(absolute_path=EVAL_DATASET_PATH)
    print(dataset)