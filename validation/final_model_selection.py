#final_model_selection.py

import json
from pathlib import Path

import pandas as pd

from shared.config import ML_OUTPUT_DIR, PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"

EXPERIMENTS_PATH = ML_OUTPUT_DIR / "experiment_results.csv"
CALIBRATED_LGBM_PATH = (
    ML_OUTPUT_DIR
    / "calibration"
    / "calibrated_lightgbm"
    / "calibrated_lightgbm_results.csv"
)

OUTPUT_CSV = OUTPUT_DIR / "final_model_selection.csv"
OUTPUT_JSON = OUTPUT_DIR / "final_model_selection.json"
OUTPUT_MD = OUTPUT_DIR / "final_model_selection.md"


def load_finalists():
    experiments = pd.read_csv(EXPERIMENTS_PATH)
    calibrated = pd.read_csv(CALIBRATED_LGBM_PATH)

    logistic = experiments[
        (experiments["model"] == "LogisticRegression")
        & (experiments["feature_set"] == "engineered")
    ].iloc[0]

    lightgbm = calibrated[
        (calibrated["feature_set"] == "engine")
        & (calibrated["calibration_method"] == "sigmoid")
    ].iloc[0]

    return logistic, lightgbm


def build_comparison(logistic, lightgbm):
    rows = [
        {
            "metric": "Hold-out Accuracy",
            "logistic_regression": logistic["accuracy"],
            "calibrated_lightgbm": lightgbm["accuracy"],
            "winner": "Calibrated LightGBM",
            "priority": "secondary",
        },
        {
            "metric": "CV Accuracy",
            "logistic_regression": logistic["cv_accuracy_mean"],
            "calibrated_lightgbm": None,
            "winner": "Logistic Regression",
            "priority": "informational",
        },
        {
            "metric": "Hold-out Log Loss",
            "logistic_regression": logistic["log_loss"],
            "calibrated_lightgbm": lightgbm["log_loss"],
            "winner": "Calibrated LightGBM",
            "priority": "primary",
        },
        {
            "metric": "CV Log Loss",
            "logistic_regression": logistic["cv_log_loss_mean"],
            "calibrated_lightgbm": None,
            "winner": "Logistic Regression",
            "priority": "primary",
        },
        {
            "metric": "Hold-out Brier Score",
            "logistic_regression": logistic["brier_score"],
            "calibrated_lightgbm": lightgbm["brier_score"],
            "winner": "Calibrated LightGBM",
            "priority": "primary",
        },
        {
            "metric": "CV Log Loss Std",
            "logistic_regression": logistic["cv_log_loss_std"],
            "calibrated_lightgbm": None,
            "winner": "Logistic Regression",
            "priority": "stability",
        },
        {
            "metric": "Training Time Seconds",
            "logistic_regression": logistic["train_seconds"],
            "calibrated_lightgbm": None,
            "winner": "Logistic Regression",
            "priority": "efficiency",
        },
        {
            "metric": "Model Complexity",
            "logistic_regression": "Low",
            "calibrated_lightgbm": "Medium",
            "winner": "Logistic Regression",
            "priority": "efficiency",
        },
        {
            "metric": "Calibration Method",
            "logistic_regression": "Native logistic probabilities",
            "calibrated_lightgbm": "Sigmoid calibration",
            "winner": "Tie",
            "priority": "primary",
        },
    ]

    return pd.DataFrame(rows)


def write_markdown(comparison_df):
    markdown = f"""# Final Model Selection — World Cup 2026 Predictor

## Finalists

### Candidate A: Tuned Logistic Regression

- Feature set: `engineered`
- Calibration: native logistic probabilities
- Strengths: probability quality, stability, speed, simplicity

### Candidate B: Calibrated LightGBM

- Feature set: `engine`
- Calibration: sigmoid
- Strengths: hold-out accuracy, nonlinear modeling capacity, calibrated probability quality

## Head-to-Head Comparison

{comparison_df.to_markdown(index=False)}

## Final Recommendation

**Recommended production model: Calibrated LightGBM with sigmoid calibration using the `engine` feature set.**

## Rationale

Tuned Logistic Regression remains the strongest simple baseline. It is fast, stable, interpretable, and has excellent cross-validation probability quality.

However, after sigmoid calibration, LightGBM reaches comparable or slightly better hold-out probability quality while preserving stronger classification performance. Its hold-out log loss and Brier score are slightly better than the tuned Logistic Regression finalist, and its hold-out accuracy is substantially higher.

Because the World Cup simulator depends on probability estimates rather than only hard class predictions, calibration remains the most important concern. The calibrated LightGBM result is strong enough to justify selecting it as the current production candidate, while keeping tuned Logistic Regression as the main fallback and benchmark.

## Caveat

The calibrated LightGBM result should eventually be validated with cross-validated calibration. Until then, the recommendation is strong but provisional.

## Next Phase

Proceed to Phase 6: Production Integration.

The next task is to build an ML match inference module that loads the selected model, prepares match features, outputs calibrated probabilities, and feeds those probabilities into the Monte Carlo World Cup simulator.
"""

    OUTPUT_MD.write_text(markdown, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logistic, lightgbm = load_finalists()
    comparison_df = build_comparison(logistic, lightgbm)

    comparison_df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison_df.to_dict(orient="records"), f, indent=2)

    write_markdown(comparison_df)

    print("Final model selection complete.")
    print(comparison_df.to_string(index=False))
    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()