#General
import os
import sys
import json
import logging
from typing import Dict, Any
import unittest
from dotenv import load_dotenv
from ruamel.yaml import YAML
from datasets import Dataset

from llm_provider import get_llm
from excel_writer import write_results_to_excel

#Ragas
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.metrics import FactualCorrectness, AnswerSimilarity

from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate

#Embeddings & VectorStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        
        embeddings_llm = get_llm(config["embeddings"], type="embeddings")
        self.embeddings_llm = LangchainEmbeddingsWrapper(embeddings_llm)

        # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
        prompt = hub.pull("rlm/rag-prompt")

        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=self.embeddings_llm)
     
        self.llm_to_be_evaluated = get_llm(self.config["llm_to_be_evaluated"])
        
        self.chain = (
            {
                "context": vectorstore.as_retriever() | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm_to_be_evaluated
            | StrOutputParser()
        )

        self.ragas_helper_llm = get_llm(self.config["ragas_helper_llm"])
        self.ragas_helper_llm = LangchainLLMWrapper(self.ragas_helper_llm)
        self.init_ragas_metrics()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def init_ragas_metrics(self):

        self.metrics = []
        metric = FactualCorrectness()
        # metric = AnswerSimilarity()
        self.metrics += [metric]

        for metric in self.metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = self.ragas_helper_llm
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = self.embeddings_llm

            run_config = RunConfig()
            metric.init(run_config) 
            
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
    
    def run_experiment(self):
        evaluation_batch = self.get_evaluation_batch()
        ds = Dataset.from_dict(evaluation_batch)
        results = evaluate(ds, metrics=self.metrics)
        logger.info(f"Final scores {results}")
        write_results_to_excel(results, self.metrics, self.config)
     
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