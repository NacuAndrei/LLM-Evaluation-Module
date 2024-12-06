import os
import sys
import json
import logging
from typing import List, Dict, Any

import unittest
import pandas as pd
from dotenv import load_dotenv
from ruamel.yaml import YAML
from datasets import Dataset

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.llms import OpenAIChat

from ragas.metrics import faithfulness, answer_similarity, context_precision
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas import evaluate
from langchain_core.output_parsers import StrOutputParser

from custom_ragas_metric_prompt import CUSTOM_LONG_FORM_ANSWER_PROMPT

load_dotenv()

EVAL_DATASET_PATH = os.environ.get('DATASET_FILENAME')
CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        embeddings = OpenAIEmbeddings(model=self.config["embeddings"]["model"])
        self.embedding = LangchainEmbeddingsWrapper(embeddings)

        llm = OpenAIChat(model=self.config["eval_llm"]["model"])
        self.llm = LangchainLLMWrapper(llm)        

        self.init_ragas_metrics()
    
    def get_available_metrics(self):
        metrics = {
            "faithfulness": faithfulness,
            "answer_similarity": answer_similarity,
            "context_precision": context_precision,
        }
        metrics["faithfulness"].statement_prompt = CUSTOM_LONG_FORM_ANSWER_PROMPT
        return metrics

    def init_ragas_metrics(self):
        available_metrics = self.get_available_metrics()
        self.metrics = []
        for name in self.config["test"]["metrics"]:
            try:
                self.metrics += [available_metrics[name]]
            except AttributeError:
                logger.warning(f"Skipping metric {name} - currently not available")

        for metric in self.metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = self.llm
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = self.embedding
            run_config = RunConfig()
            metric.init(run_config)
            
    def load_dataset(self, absolute_path: str):
        with open(absolute_path, "r") as json_file:
            all_examples = json.load(json_file)
        return all_examples
    
    def get_chunks_from_chain(self, response):
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # chunks = text_splitter.split_documents(documents)
        chunks = response["sources"].split("\n\n\n")[1:]
        for idx, chunk in enumerate(chunks):
            chunks[idx] = chunk.split("is below.\n")[1]
        logger.debug(f"Contexts: \n {chunks}")
        return chunks

    def score_with_ragas(self, query: str, ground_truth: str, chunks: List[str], answer: str):
        scores = {}
        for m in self.metrics:
            logger.debug(f"calculating: {m.name}")
            scores[m.name] = m.score(
                row={
                    "question": query,
                    "ground_truth": ground_truth,
                    "context": chunks,
                    "answer": answer,
                }
            )
            logger.info(f"{m.name} score: {scores[m.name]}")
        return scores
    
    def get_complete_example(self, example: Dict[str, Any]):
        question = example["question"]
        prompt = f"Please formulate the answer for the following question in one or multiple sentences. This is the question: {question}"

        self.chain = prompt | self.llm | StrOutputParser()
        response = self.chain.invoke(prompt)
        
        example["model_answer"] = response["response"]                

        return example

    def get_evaluation_batch(self):
        evaluation_batch = {
            "question": [],
            "ground_truth": [],
            "model_answer": []
        }

        # Ingest relevant documents for questions
        # self.chain.add_files_to_store(directory=os.environ["EVAL_SOURCE_DOC_DIR"])

        #-----USE LANGCHAIN READER TO LOAD THE DOCUMENTS-----
        # # Take everything in all the sub-folders of our knowledgebase
        
        # folders = glob.glob("knowledge-base/*")

        # text_loader_kwargs = {'encoding': 'utf-8'}
        # # If that doesn't work, some Windows users might need to uncomment the next line instead
        # # text_loader_kwargs={'autodetect_encoding': True}

        # documents = []
        # for folder in folders:
        #     doc_type = os.path.basename(folder)
        #     loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        #     folder_docs = loader.load()
        #     for doc in folder_docs:
        #         doc.metadata["doc_type"] = doc_type
        #     documents.append(doc)
        
        self.incomplete_examples = self.load_dataset(absolute_path=EVAL_DATASET_PATH)

        for incomplete_example in self.incomplete_examples:
            complete_example = self.get_complete_example(incomplete_example)

            evaluation_batch["question"].append(complete_example["question"])
            evaluation_batch["ground_truth"].append(complete_example["ground_truth"])
            evaluation_batch["model_answer"].append(complete_example["model_answer"])

        return evaluation_batch
    
    def write_results_to_excel(self, results):
        new_data = {
            "Doc format": [self.config["test"]["doc_format"]],
            "# QA GT": [len(self.all_examples)],
            "top_k": [self.chain_config["retriever"]["search_kwargs"]["k"]],
            "chunk_size": [CHUNK_SIZES[self.chain_config["ingestion"]["chunk_size"]]],
            "chunk_overlap": [self.chain_config["ingestion"]["overlap"]],
            "embed_model": [self.embed_model_id],
            "llm": [self.chain_config["llm"]["model_id"]],
            "eval llm": [self.config["eval_llm"]["model_id"]],
        }
        for metric in self.metrics:
            new_data[metric.name] = results[metric.name]
        df_new = pd.DataFrame(new_data)
        df_existing = pd.read_excel(os.environ["RESULTS_EXCEL_PATH"])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(os.environ["RESULTS_EXCEL_PATH"], index=False)
        
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
    unittest.main(exit=False)