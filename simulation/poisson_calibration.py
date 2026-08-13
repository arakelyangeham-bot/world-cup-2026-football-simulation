#poisson_calibration.py

from pathlib import Path
import csv
from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COEFFICIENTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_coefficients.csv"
)

@lru_cache(maxsize=1)
def load_poisson_goal_coefficients(
    path: Path = COEFFICIENTS_FILE,
) -> dict[str, dict[str, float]]:
    """
    Load fitted Poisson goal model coefficients.

    Returns:
        {
            "home_goal_model": {
                "intercept": ...,
                "home_poisson_attack": ...,
                ...
            },
            "away_goal_model": {
                "intercept": ...,
                "away_poisson_attack": ...,
                ...
            },
        }
    """
    coefficients: dict[str, dict[str, float]] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            model = row["model"]
            feature = row["feature"]
            value = float(row["coefficient"])

            if model not in coefficients:
                coefficients[model] = {}

            coefficients[model][feature] = value

    return coefficients