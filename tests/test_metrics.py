import os
import sys
import logging

from dotenv import load_dotenv

EVAL_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), f"data/{os.environ.get('DATASET_FILENAME')}"
)

CHUNK_SIZES = {"small": 300, "medium": 650, "large": 1000}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_available_metrics():
    metrics = {}
    
    return metrics

