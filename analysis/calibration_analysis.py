# calibration_analysis.py

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from shared.calibration import multiclass_brier_score
from shared.config import ML_OUTPUT_DIR


CALIBRATION_DIR = ML_OUTPUT_DIR / "calibration"
REGISTRY_PATH = ML_OUTPUT_DIR / "experiment_results.csv"


def load_predictions(model_name, feature_set):
    model_dir_map = {
        "LogisticRegression": "logistic",
        "LightGBM": "lightgbm",
        "XGBoost": "xgboost",
    }

    file_prefix_map = {
        "LogisticRegression": "logisticregression",
        "LightGBM": "lightgbm",
        "XGBoost": "xgboost",
    }

    model_dir = model_dir_map[model_name]
    file_prefix = file_prefix_map[model_name]

    path = (
        ML_OUTPUT_DIR
        / model_dir
        / f"{file_prefix}_{feature_set}_predictions.csv"
    )

    return pd.read_csv(path)


def get_probability_columns(predictions_df):
    return [
        col
        for col in predictions_df.columns
        if col.startswith("prob_")
    ]


def evaluate_prediction_file(model_name, feature_set):
    predictions_df = load_predictions(model_name, feature_set)

    prob_cols = get_probability_columns(predictions_df)

    labels = [
        col.replace("prob_", "")
        for col in prob_cols
    ]

    if all(label.isdigit() for label in labels):
        labels = [int(label) for label in labels]
    
    y_true = predictions_df["actual"]
    y_proba = predictions_df[prob_cols].values
    y_proba = predictions_df[prob_cols].values
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    metrics = {
        "model": model_name,
        "feature_set": feature_set,
        "log_loss": log_loss(y_true, y_proba, labels=labels),
        "brier_score": multiclass_brier_score(y_true, y_proba, labels),
    }

    return predictions_df, labels, metrics


def plot_calibration(predictions_df, labels, title, output_path):
    plt.figure(figsize=(8, 6))

    for label in labels:
        y_binary = (predictions_df["actual"] == label).astype(int)
        y_prob = predictions_df[f"prob_{label}"]

        frac_pos, mean_pred = calibration_curve(
            y_binary,
            y_prob,
            n_bins=5,
            strategy="uniform",
        )

        plt.plot(mean_pred, frac_pos, marker="o", label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("LogisticRegression", "engineered"),
        ("LogisticRegression", "raw"),
        ("LightGBM", "raw"),
        ("LightGBM", "engine"),
        ("XGBoost", "engine"),
    ]

    rows = []

    for model_name, feature_set in experiments:
        predictions_df, labels, metrics = evaluate_prediction_file(
            model_name,
            feature_set,
        )

        rows.append(metrics)

        plot_path = (
            CALIBRATION_DIR
            / f"{model_name.lower()}_{feature_set}_calibration.png"
        )

        plot_calibration(
            predictions_df=predictions_df,
            labels=labels,
            title=f"{model_name} ({feature_set}) Calibration",
            output_path=plot_path,
        )

        print(f"Saved calibration plot: {plot_path}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = CALIBRATION_DIR / "calibration_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_log_loss = metrics_df.sort_values("log_loss").iloc[0]
    best_brier = metrics_df.sort_values("brier_score").iloc[0]

    summary = f"""# Calibration Analysis

## Summary

| Metric | Best Model | Feature Set | Value |
|---|---|---|---:|
| Log Loss | {best_log_loss['model']} | {best_log_loss['feature_set']} | {best_log_loss['log_loss']:.4f} |
| Brier Score | {best_brier['model']} | {best_brier['feature_set']} | {best_brier['brier_score']:.4f} |

## Interpretation

Lower log loss and lower Brier score indicate better probability quality.

For the World Cup simulator, the best-calibrated model is more important than the model with the highest raw classification accuracy.
"""

    summary_path = CALIBRATION_DIR / "calibration_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(f"Saved calibration metrics: {metrics_path}")
    print(f"Saved calibration summary: {summary_path}")
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()