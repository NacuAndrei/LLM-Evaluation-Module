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
    
    def generate_single_trace(self, example: Dict[str, Any], trace_name: str = "rag"):
        question = example["question"]
        ground_truth = example["ground_truth"]
        trace = self.langfuse.trace(name=trace_name)

        prompt = f"Please formulate the answer for the following question in one or multiple sentences. This is the question: {question}"
        response = self.chain.invoke(prompt)

        contexts = self.get_chunks_from_chain(response)
        trace.span(
            name="retrieval",
            input={"question": question},
            output={"contexts": contexts},
        )

        answer = self.get_response_from_chain(response)

        trace.span(
            name="generation",
            input={
                "question": question,
                "contexts": contexts,
                "ground_truth": ground_truth,
            },
            output={"answer": answer},
        )
        
    def generate_traces(self, trace_name: str = "rag"):
        self.chain.add_files_to_store(directory=os.environ["EVAL_SOURCE_DOC_DIR"])

        self.all_examples = self.load_dataset(absolute_path=EVAL_DATASET_PATH)
        for example in self.all_examples:
            self.generate_single_trace(example, trace_name)
            
    def get_traces(self, name="rag", limit=None, user_id=None):
        all_data = []
        page = 1

        while True:
            response = self.langfuse.client.trace.list(
                name=name, page=page, user_id=user_id
            )
            if not response.data:
                break
            page += 1
            all_data.extend(response.data)
            if limit is not None and len(all_data) > limit:
                break
        if limit is None:
            return all_data
        return all_data[:limit]