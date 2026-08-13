#calibrate_outcome_probabilities.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "probability_calibration_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "outcome_probability_calibration_experiments.csv"
)

OUTCOME_ORDER = ["home_win", "draw", "away_win"]


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))


def expected_calibration_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(observed)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predicted >= bins[i]) & (predicted <= bins[i + 1])
        else:
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])

        if not np.any(mask):
            continue

        observed_frequency = observed[mask].mean()
        predicted_mean = predicted[mask].mean()
        ece += (mask.sum() / total) * abs(observed_frequency - predicted_mean)

    return ece


def mean_ece(df: pd.DataFrame, probs: np.ndarray) -> float:
    return float(
        np.mean(
            [
                expected_calibration_error(
                    (df["outcome"] == "home_win").to_numpy(dtype=int),
                    probs[:, 0],
                ),
                expected_calibration_error(
                    (df["outcome"] == "draw").to_numpy(dtype=int),
                    probs[:, 1],
                ),
                expected_calibration_error(
                    (df["outcome"] == "away_win").to_numpy(dtype=int),
                    probs[:, 2],
                ),
            ]
        )
    )


def draw_temperature_adjustment(
    probs: np.ndarray,
    draw_multiplier: float,
) -> np.ndarray:
    adjusted = probs.copy()
    adjusted[:, 1] *= draw_multiplier
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted

def class_multiplier_adjustment(
    probs: np.ndarray,
    home_multiplier: float,
    draw_multiplier: float,
    away_multiplier: float,
) -> np.ndarray:
    multipliers = np.array(
        [
            home_multiplier,
            draw_multiplier,
            away_multiplier,
        ],
        dtype=float,
    )

    adjusted = probs * multipliers
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

    return adjusted


def evaluate(
    df: pd.DataFrame,
    model_name: str,
    probs: np.ndarray,
    parameters: str,
) -> dict[str, object]:
    outcome_to_index = {outcome: idx for idx, outcome in enumerate(OUTCOME_ORDER)}

    y_true = df["outcome"].map(outcome_to_index).to_numpy(dtype=int)

    return {
        "model": model_name,
        "parameters": parameters,
        "matches": len(df),
        "multiclass_brier_score": multiclass_brier_score(y_true, probs),
        "multiclass_log_loss": log_loss(y_true, probs, labels=[0, 1, 2]),
        "ece_mean": mean_ece(df, probs),
    }


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    base_probs = df[
        [
            "p_home_win",
            "p_draw",
            "p_away_win",
        ]
    ].to_numpy(dtype=float)

    rows = []

    rows.append(
        evaluate(
            df=df,
            model_name="identity",
            probs=base_probs,
            parameters="none",
        )
    )

    for home_multiplier in np.round(np.arange(0.95, 1.051, 0.01), 2):
        for draw_multiplier in np.round(np.arange(0.90, 1.211, 0.01), 2):
            for away_multiplier in np.round(np.arange(0.95, 1.051, 0.01), 2):

                adjusted_probs = class_multiplier_adjustment(
                    base_probs,
                    home_multiplier=float(home_multiplier),
                    draw_multiplier=float(draw_multiplier),
                    away_multiplier=float(away_multiplier),
                )

                rows.append(
                    evaluate(
                        df=df,
                        model_name="class_multiplier",
                        probs=adjusted_probs,
                        parameters=(
                            f"home={home_multiplier:.2f};"
                            f"draw={draw_multiplier:.2f};"
                            f"away={away_multiplier:.2f}"
                        ),
                    )
                )

    results = pd.DataFrame(rows).sort_values(
        [
            "multiclass_log_loss",
            "multiclass_brier_score",
            "ece_mean",
        ]
    )

    baseline = results[results["model"] == "identity"].iloc[0]

    results["delta_brier"] = (
        results["multiclass_brier_score"]
        - baseline["multiclass_brier_score"]
    )

    results["delta_log_loss"] = (
        results["multiclass_log_loss"]
        - baseline["multiclass_log_loss"]
    )

    results["delta_ece_mean"] = (
        results["ece_mean"]
        - baseline["ece_mean"]
    )

    results = results.sort_values(
        [
            "delta_log_loss",
            "delta_ece_mean",
            "delta_brier",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Outcome Probability Calibration Experiments")
    print("-------------------------------------------")
    print(results.head(15).to_string(index=False))
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()