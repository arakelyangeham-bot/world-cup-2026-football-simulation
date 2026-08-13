#audit_prediction_pipeline.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inference.model_paths import PRODUCTION_MODEL_PATH
from simulation.probability_adapter import ProbabilityAdapter
from shared.dataset import prepare_train_test_data
from shared.feature_sets import get_feature_set
from shared.label_encoding import encode_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "prediction_pipeline_audit.csv"
)


def row_to_team_dict(row: pd.Series, prefix: str) -> dict[str, float]:
    return {
        "attack": row[f"{prefix}_attack"],
        "midfield": row[f"{prefix}_midfield"],
        "defense": row[f"{prefix}_defense"],
        "gk": row[f"{prefix}_gk"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def main() -> None:
    features = get_feature_set("v2")

    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    (
        y_encoded,
        y_train_encoded,
        y_test_encoded,
        encoded_labels,
        label_mapping,
        label_encoder,
    ) = encode_labels(y, y_train, y_test)

    model = joblib.load(PRODUCTION_MODEL_PATH)
    direct_probs = model.predict_proba(X)

    adapter = ProbabilityAdapter(mode="ml")

    historical_path = (
        PROJECT_ROOT
        / "outputs"
        / "model_training"
        / "historical_training_dataset.csv"
    )

    source_df = pd.read_csv(historical_path)
    rows = []

    for idx in range(min(20, len(source_df))):
        row = source_df.iloc[idx]

        home_team = row_to_team_dict(row, "home")
        away_team = row_to_team_dict(row, "away")

        adapter_probs = adapter.predict_match(home_team, away_team)

        direct = direct_probs[idx]

        rows.append(
            {
                "row_index": idx,
                "actual_encoded": int(y_encoded[idx]),
                "actual_label": label_mapping[int(y_encoded[idx])],

                "direct_class_0": direct[0],
                "direct_class_1": direct[1],
                "direct_class_2": direct[2],

                "adapter_away_win": adapter_probs["away_win"],
                "adapter_draw": adapter_probs["draw"],
                "adapter_home_win": adapter_probs["home_win"],

                "diff_class0_vs_away": direct[0] - adapter_probs["away_win"],
                "diff_class1_vs_draw": direct[1] - adapter_probs["draw"],
                "diff_class2_vs_home": direct[2] - adapter_probs["home_win"],

                "direct_pred_encoded": int(np.argmax(direct)),
                "adapter_pred_label": max(
                    adapter_probs,
                    key=adapter_probs.get,
                ),
            }
        )

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Prediction Pipeline Audit")
    print("-------------------------")
    print(output.to_string(index=False))
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()