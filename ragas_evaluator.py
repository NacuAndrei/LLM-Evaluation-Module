import logging
import os

from datasets import Dataset

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate

from chain_manager import ChainManager
from metric_initializer import MetricInitializer
from data_loader import DataLoader
from llm_provider import get_llm
from excel_writer import write_results_to_excel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config

        self.embeddings_llm = get_llm(config["embeddings"], type="embeddings")
        #self.embeddings_llm = LangchainEmbeddingsWrapper(embeddings_llm)

        self.chain_manager = ChainManager(config, self.embeddings_llm)
        self.chain = self.chain_manager.setup_chain()

        ragas_helper_llm = get_llm(config["ragas_helper_llm"])
        self.ragas_helper_llm = LangchainLLMWrapper(ragas_helper_llm)

        self.metric_initializer = MetricInitializer(config)
        self.metrics = self.metric_initializer.init_metrics(self.embeddings_llm, self.ragas_helper_llm)

    def get_complete_sample(self, sample: dict):
        question = sample["question"]
        prompt = f"Please formulate the answer for the following question in one or multiple sentences. This is the question: {question}"
        
        logger.info(f"Generated prompt: {prompt}")
        
        try:
            response = self.chain.invoke(prompt)
            sample["response"] = response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            sample["response"] = "Error generating response"

        return sample

    def get_evaluation_ds(self):
        evaluation_batch = {
            "question": [],
            "ground_truth": [],
            "response": []
        }

        try:
            self.incomplete_samples = DataLoader.load_dataset(absolute_path=os.environ.get('DATASET_FILENAME'))
            logger.info(f"Loaded {len(self.incomplete_samples)} samples from dataset.")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return evaluation_batch

        for sample in self.incomplete_samples:
            complete_sample = self.get_complete_sample(sample)
            evaluation_batch["question"].append(complete_sample["question"])
            evaluation_batch["ground_truth"].append(complete_sample["ground_truth"])
            evaluation_batch["response"].append(complete_sample["response"])

        ds = Dataset.from_dict(evaluation_batch)
        return ds

    def run_experiment(self):
        evaluation_ds = self.get_evaluation_ds()
        results = evaluate(evaluation_ds, metrics=self.metrics) #Should we give the llm and embeddings as arguments?
        logger.info(f"Final scores {results}")
        write_results_to_excel(results, self.metrics, self.config)
