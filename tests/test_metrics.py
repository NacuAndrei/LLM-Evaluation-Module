import os
import sys
import json
import logging
import time
import pandas as pd
from typing import List, Dict, Any

from dotenv import load_dotenv
from langfuse import Langfuse
from datasets import Dataset
from ragas.metrics import faithfulness, answer_similarity, context_precision
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig
from ragas import evaluate

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
    
    def write_results_to_excel(self, results):
        new_data = {
            "Doc format": [self.test_config["test"]["doc_format"]],
            "# QA GT": [len(self.all_examples)],
            "top_k": [self.chain_config["retriever"]["search_kwargs"]["k"]],
            "chunk_size": [CHUNK_SIZES[self.chain_config["ingestion"]["chunk_size"]]],
            "chunk_overlap": [self.chain_config["ingestion"]["overlap"]],
            "embed_model": [self.embed_model_id],
            "llm": [self.chain_config["llm"]["model_id"]],
            "eval llm": [self.test_config["eval_llm"]["model_id"]],
        }
        for metric in self.metrics:
            new_data[metric.name] = results[metric.name]
        df_new = pd.DataFrame(new_data)
        df_existing = pd.read_excel(os.environ["RESULTS_EXCEL_PATH"])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(os.environ["RESULTS_EXCEL_PATH"], index=False)
        
    def evaluate_batch(self, trace_name: str = "rag"):
        traces = self.get_traces(name=trace_name)
        evaluation_batch = {
            "question": [],
            "contexts": [],
            "ground_truth": [],
            "answer": [],
            "trace_id": [],
        }

        for t in traces:
            observations = [
                self.langfuse.client.observations.get(o) for o in t.observations
            ]
            for o in observations:
                if o.name == "retrieval":
                    question = o.input["question"]
                    contexts = o.output["contexts"]
                if o.name == "generation":
                    ground_truth = o.input["ground_truth"]
                    answer = o.output["answer"]
            evaluation_batch["question"].append(question)
            evaluation_batch["contexts"].append(contexts)
            evaluation_batch["trace_id"].append(t.id)
            evaluation_batch["ground_truth"].append(ground_truth)
            evaluation_batch["answer"].append(answer)

        ds = Dataset.from_dict(evaluation_batch)
        r = evaluate(ds, metrics=self.metrics)
        logger.info(f"Final scores {r}")
        self.write_results_to_excel(results=r)

        df = r.to_pandas()

        # add the langfuse trace_id to the result dataframe
        df["trace_id"] = ds["trace_id"]

        df.head()

        for _, row in df.iterrows():
            for metric in self.metrics:
                if pd.isna(row[metric.name]):
                    row[metric.name] = 0
                    logger.warning(
                        f"Evaluation model output could not be parsed, affected trace: {row['trace_id']} metric: {metric.name}."
                    )
                logger.info(
                    f"Trace {row['trace_id']} metric {metric.name}: {row[metric.name]}"
                )
                self.langfuse.score(
                    name=metric.name, value=row[metric.name], trace_id=row["trace_id"]
                )
    
    def overwrite_with_test_config(self, default_config, test_config):
        chain_config = default_config.copy()
        overwrite_keys = [
            key for key in test_config.keys() if key not in ["eval_llm", "test"]
        ]
        for chain_key in overwrite_keys:
            for attribute in test_config[chain_key].keys():
                chain_config[chain_key][attribute] = test_config[chain_key][attribute]

        available_metrics = get_available_metrics()
        self.metrics = []
        for name in test_config["test"]["metrics"]:
            try:
                self.metrics += [available_metrics[name]]
            except AttributeError:
                logger.warning(f"Skipping metric {name} - currently not available")
        return chain_config

    # def create_chain(self, chain_config):
    #     self.chain = RAGChadChain.from_config(chain_config)

    def run_experiment(self, generate_traces: bool = True, trace_name: str = "rag"):
        if generate_traces:
            self.generate_traces(trace_name)
            self.langfuse.flush()
            # Wait for all traces to be created
            # Langfuse_flush should block execution until all events are sent
            # However sometimes the traces are still not fully created
            time.sleep(10)
        self.evaluate_batch(trace_name)