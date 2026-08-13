#production_recommendation.py

import json

import pandas as pd

from shared.config import ML_OUTPUT_DIR


EXPERIMENTS_PATH = ML_OUTPUT_DIR / "experiment_results.csv"
CALIBRATED_LGBM_PATH = (
    ML_OUTPUT_DIR
    / "calibration"
    / "calibrated_lightgbm"
    / "calibrated_lightgbm_results.csv"
)

OUTPUT_CSV = ML_OUTPUT_DIR / "production_recommendation.csv"
OUTPUT_MD = ML_OUTPUT_DIR / "production_recommendation.md"
OUTPUT_JSON = ML_OUTPUT_DIR / "production_recommendation.json"


def normalize_lower_is_better(series):
    return 1 - ((series - series.min()) / (series.max() - series.min()))


def normalize_higher_is_better(series):
    return (series - series.min()) / (series.max() - series.min())


def build_candidates():
    experiments = pd.read_csv(EXPERIMENTS_PATH)
    calibrated_lgbm = pd.read_csv(CALIBRATED_LGBM_PATH)

    logistic = experiments[
        (experiments["model"] == "LogisticRegression")
        & (experiments["feature_set"] == "engineered")
    ].iloc[0]

    lightgbm_cal = calibrated_lgbm[
        (calibrated_lgbm["feature_set"] == "engine")
        & (calibrated_lgbm["calibration_method"] == "sigmoid")
    ].iloc[0]

    candidates = pd.DataFrame(
        [
            {
                "candidate": "Tuned Logistic Regression",
                "model": "LogisticRegression",
                "feature_set": "engineered",
                "calibration": "native",
                "accuracy": logistic["accuracy"],
                "cv_accuracy": logistic["cv_accuracy_mean"],
                "log_loss": logistic["log_loss"],
                "cv_log_loss": logistic["cv_log_loss_mean"],
                "brier_score": logistic["brier_score"],
                "cv_log_loss_std": logistic["cv_log_loss_std"],
                "train_seconds": logistic["train_seconds"],
                "complexity_score": 1,
            },
            {
                "candidate": "Calibrated LightGBM",
                "model": "LightGBM",
                "feature_set": "engine",
                "calibration": "sigmoid",
                "accuracy": lightgbm_cal["accuracy"],
                "cv_accuracy": None,
                "log_loss": lightgbm_cal["log_loss"],
                "cv_log_loss": None,
                "brier_score": lightgbm_cal["brier_score"],
                "cv_log_loss_std": None,
                "train_seconds": None,
                "complexity_score": 3,
            },
        ]
    )

    return candidates


def score_candidates(candidates):
    scored = candidates.copy()

    scored["score_log_loss"] = normalize_lower_is_better(scored["log_loss"])
    scored["score_brier"] = normalize_lower_is_better(scored["brier_score"])
    scored["score_accuracy"] = normalize_higher_is_better(scored["accuracy"])
    scored["score_complexity"] = normalize_lower_is_better(
        scored["complexity_score"]
    )

    scored["production_score"] = (
        0.40 * scored["score_log_loss"]
        + 0.30 * scored["score_brier"]
        + 0.20 * scored["score_accuracy"]
        + 0.10 * scored["score_complexity"]
    )

    return scored.sort_values("production_score", ascending=False)


def write_report(scored):
    winner = scored.iloc[0]
    runner_up = scored.iloc[1]

    markdown = f"""# World Cup 2026 Production Model Recommendation

## Recommended Production Candidate

**{winner['candidate']}**

## Candidate Comparison

{scored.to_markdown(index=False)}

## Interpretation

The recommendation uses a weighted score:

- 40% hold-out log loss
- 30% Brier score
- 20% hold-out accuracy
- 10% model simplicity

The selected model is **{winner['candidate']}** with the **{winner['feature_set']}** feature set.

The runner-up is **{runner_up['candidate']}**.

## Practical Recommendation

Use **{winner['candidate']}** as the current production candidate for the World Cup simulator.

Continue tracking **{runner_up['candidate']}** as the main challenger, especially if future cross-validated calibration improves its probability estimates.

## Notes

- Log loss and Brier score are prioritized because the simulator depends on calibrated probabilities.
- Accuracy matters, but it should not dominate model selection.
- The current recommendation should be revisited after cross-validated calibration and ensemble testing.
"""

    OUTPUT_MD.write_text(markdown, encoding="utf-8")


def main():
    candidates = build_candidates()
    scored = score_candidates(candidates)

    scored.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(scored.to_dict(orient="records"), f, indent=2)

    write_report(scored)

    print("Production recommendation complete.")
    print(scored.to_string(index=False))
    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Markdown: {OUTPUT_MD}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()