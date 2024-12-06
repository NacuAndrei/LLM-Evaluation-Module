#General
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

#Ragas
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate

#Embeddings & VectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

#LLM To be Evaluated
from langchain_community.llms import OpenAIChat
#from langchain_openai.chat_models import ChatOpenAI

#Temporary
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

EVAL_DATASET_PATH = os.environ.get('DATASET_FILENAME')
CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        embeddings = OpenAIEmbeddings(model=self.config["embeddings"]["model"])
        self.embedding = LangchainEmbeddingsWrapper(embeddings)
                
        self.llm_to_be_evaluated = OpenAIChat(model=self.config["eval_llm"]["model"])            

        # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
        prompt = hub.pull("rlm/rag-prompt")

        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()

        # Split text in chunks and store it in FAISS vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

        #llm_to_be_evaluated Chain Creation
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
        self.llm = LangchainLLMWrapper(self.llm_to_be_evaluated)
        self.init_ragas_metrics()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def init_ragas_metrics(self):
        self.metrics = []

        metric = FactualCorrectness() #TODO 2 Add more metrics
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

        response = self.chain.invoke(prompt) #TODO 1 find a working pair of versions for Ragas & OpenAI (Ragas 0.2.7 calls openai.ChatCompletion, but this is no longer supported in openai>=1.0.0; but ragas 0.2.7 depends on openai>1; ???)
        
        sample["model_answer"] = response["response"]                

        return sample

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
        
        self.incomplete_samples = self.load_dataset(absolute_path=EVAL_DATASET_PATH)

        for incomplete_sample in self.incomplete_samples:
            complete_sample = self.get_complete_sample(incomplete_sample)

            evaluation_batch["question"].append(complete_sample["question"])
            evaluation_batch["ground_truth"].append(complete_sample["ground_truth"])
            evaluation_batch["model_answer"].append(complete_sample["model_answer"])

        return evaluation_batch
    
    def write_results_to_excel(self, results):
        new_data = {
            "Doc format": [self.config["test"]["doc_format"]],
            "# QA GT": [len(self.all_samples)],
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