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
from langchain_openai import ChatOpenAI

import openai

from ragas.metrics import faithfulness, answer_similarity, context_precision
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas import evaluate
#from langchain.document_loaders import DirectoryLoader, TextLoader

load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai.api_key = os.getenv("OPENAI_API_KEY")

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), f"data/{os.environ.get('DATASET_FILENAME')}")
CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.metrics = self.initialize_metrics()
        
        # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        embeddings = OpenAIEmbeddings(model=self.config["embeddings"]["model"])
        self.embedding = LangchainEmbeddingsWrapper(embeddings)

        eval_llm = ChatOpenAI(model=self.config["eval_llm"]["model"])
        self.eval_llm = LangchainLLMWrapper(self.initialize_gpt4(eval_llm))

        # self.create_chain(self.config)
        
        #-----CREATE CHROMA VECTORSTORE-----
        # if os.path.exists(db_name):
        # Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

        # # Create vectorstore

        # vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
        # print(f"Vectorstore created with {vectorstore._collection.count()} documents")
        
        #------CREATE CHAIN----
        # llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

        # # set up the conversation memory for the chat
        # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        # # the retriever is an abstraction over the VectorStore that will be used during RAG
        # retriever = vectorstore.as_retriever()

        # # putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
        # conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        
        self.init_ragas_metrics()
    
    #----Not necessary if we are using the metrics from the config.yml----   
    # def initialize_metrics(self) -> List:
    #     return [faithfulness, answer_similarity, context_precision]
      
    def initialize_gpt4(self, config):
        return {
            "model": "gpt-4",
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 150)
        }
    
    def init_ragas_metrics(self):
        for metric in self.metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = self.eval_llm
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

    def get_response_from_chain(self, response):
        return response["answer"]
    
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
    
    def invoke_gpt4(self, prompt):
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    
    def get_complete_example(self, example: Dict[str, Any]):
        question = example["question"]
        prompt = f"Please formulate the answer for the following question in one or multiple sentences. This is the question: {question}"

        response = self.invoke_gpt4(prompt)      
        
        model_answer = self.get_response_from_chain(response)
        example["model_answer"] = model_answer

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
     
    def create_chain(self, chain_config):
        self.chain = []
        if "steps" in chain_config:
            for step in chain_config["steps"]:
                self.chain.append(step)
                logging.info(f"Step added to chain: {step}")
                
    def execute_chain(self):
        if self.chain:
            for step in self.chain:
                logging.info(f"Executing step: {step}")
                print(f"Executing: {step}")
  
class TestRagas(unittest.TestCase):

    yaml_config_file = YAML(typ="safe")
    with open(os.environ["CONFIG_PATH"], "r") as file:
        config = yaml_config_file.load(file)

    ragas_eval = RagasEvaluator(config=config)

    ragas_eval.run_experiment()


if __name__ == "__main__":
    config = {"steps": ["step1", "step2", "step3"]}
    evaluator = RagasEvaluator(config)
    evaluator.execute_chain()
