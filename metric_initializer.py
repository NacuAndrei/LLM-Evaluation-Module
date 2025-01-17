from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig

class MetricInitializer:
    def __init__(self, config: dict):
        self.config = config
        self.metrics = []

    def init_metrics(self, embeddings_llm, ragas_helper_llm):
        metric = FactualCorrectness()
        self.metrics.append(metric)

        for metric in self.metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = ragas_helper_llm
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = embeddings_llm

            run_config = RunConfig()
            metric.init(run_config)

        return self.metrics
