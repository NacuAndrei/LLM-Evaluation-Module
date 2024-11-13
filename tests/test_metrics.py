import os
import sys
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from langfuse import Langfuse
from ragas.metrics import faithfulness, answer_similarity, context_precision
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig

EVAL_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), f"data/{os.environ.get('DATASET_FILENAME')}"
)

CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_available_metrics():
    metrics = {
        "faithfulness": faithfulness,
        "answer_similarity": answer_similarity,
        "context_precision": context_precision,
    }
    
    return metrics

class RagasEvaluation:
    def __init__(self, langfuse: Langfuse, default_config: Dict[str, Any], test_config: Dict[str, Any]) -> None:
        self.langfuse = langfuse
        self.test_config = test_config
        
        if self.test_config["eval_llm"].get("provider") is not None:
            self.test_config["eval_llm"].pop("provider")
        eval_llm_config = self.test_config["eval_llm"]
        
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
        chunks = response["sources"].split("\n\n\n")[1:]
        for idx, chunk in enumerate(chunks):
            chunks[idx] = chunk.split("is below.\n")[1]
        logger.debug(f"Contexts: \n {chunks}")
        return chunks

    def get_response_from_chain(self, response):
        return response["answer"]
    
    