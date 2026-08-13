# config.py

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ML_OUTPUT_DIR = OUTPUT_DIR / "ml"

RANDOM_STATE = 42
TEST_SIZE = 0.20


HISTORICAL_TRAINING_DATASET_PATH = (
    PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"
)