import unittest
from config_loader import ConfigLoader
from ragas_evaluator import RagasEvaluator

class TestRagas(unittest.TestCase):
    config = ConfigLoader.load_config()
    ragas_eval = RagasEvaluator(config=config)

    def test_run_experiment(self):
        self.ragas_eval.run_experiment()

if __name__ == "__main__":
    unittest.main(exit=False)