#audit_outcome_model_historical_performance.py

from pathlib import Path

import pandas as pd

from inference.match_predictor import MatchPredictor
from shared.outcome_evaluation import evaluate_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    predictor = MatchPredictor()

    result = evaluate_predictor(
        predictor=predictor,
        df=df,
        model_name="production",
    )

    print("Outcome Model Historical Performance")
    print("------------------------------------")
    print(f"Rows: {result.summary['rows']}")
    print()
    print(f"Accuracy: {result.summary['accuracy']:.3f}")
    print(f"Log loss: {result.summary['log_loss']:.3f}")

    print()
    print("Actual distribution")
    print("-------------------")
    print(pd.Series(result.summary["actual_distribution"]).to_string())

    print()
    print("Predicted distribution")
    print("----------------------")
    print(pd.Series(result.summary["predicted_distribution"]).to_string())

    print()
    print("Confusion matrix")
    print("----------------")
    print(result.confusion_matrix.to_string())


if __name__ == "__main__":
    main()