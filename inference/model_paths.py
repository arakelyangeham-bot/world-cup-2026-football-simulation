# inference/model_paths.py

from shared.config import ML_OUTPUT_DIR


PRODUCTION_MODEL_DIR = ML_OUTPUT_DIR / "production"

# Canonical production artifacts.
# These always point to the current production model.
PRODUCTION_MODEL_PATH = (
    PRODUCTION_MODEL_DIR / "production_model.joblib"
)

PRODUCTION_METADATA_PATH = (
    PRODUCTION_MODEL_DIR / "production_metadata.json"
)