import os

EVAL_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), f"data/{os.environ.get('DATASET_FILENAME')}"
)

CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

def get_available_metrics():
    metrics = {}
    
    return metrics