from datasets import Dataset
from llm_provider import get_llm
from excel_writer import write_results_to_excel
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate

import logging
import os

from chain_manager import ChainManager
from metric_initializer import MetricInitializer
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config

        embeddings_llm = get_llm(config["embeddings"], type="embeddings")
        self.embeddings_llm = LangchainEmbeddingsWrapper(embeddings_llm)

        self.chain_manager = ChainManager(config, self.embeddings_llm)
        self.chain = self.chain_manager.setup_chain()

        ragas_helper_llm = get_llm(config["ragas_helper_llm"])
        self.ragas_helper_llm = LangchainLLMWrapper(ragas_helper_llm)

        self.metric_initializer = MetricInitializer(config)
        self.metrics = self.metric_initializer.init_metrics(self.embeddings_llm, self.ragas_helper_llm)

    def get_complete_sample(self, sample: dict):
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

        self.incomplete_samples = DataLoader.load_dataset(absolute_path=os.environ.get('DATASET_FILENAME'))

        for incomplete_sample in self.incomplete_samples:
            complete_sample = self.get_complete_sample(incomplete_sample)

            evaluation_batch["question"].append(complete_sample["question"])
            evaluation_batch["ground_truth"].append(complete_sample["ground_truth"])
            evaluation_batch["response"].append(complete_sample["response"])

        return evaluation_batch

    def run_experiment(self):
        evaluation_batch = self.get_evaluation_batch()
        ds = Dataset.from_dict(evaluation_batch)
        results = evaluate(ds, metrics=self.metrics) #Should we give the llm and embeddings as arguments?
        logger.info(f"Final scores {results}")
        write_results_to_excel(results, self.metrics, self.config)
