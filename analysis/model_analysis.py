# analysis/model_analysis.py

import json

import pandas as pd

from shared.config import ML_OUTPUT_DIR


REGISTRY_PATH = ML_OUTPUT_DIR / "experiment_results.csv"
ANALYSIS_CSV_PATH = ML_OUTPUT_DIR / "model_analysis.csv"
ANALYSIS_MD_PATH = ML_OUTPUT_DIR / "model_analysis.md"
ANALYSIS_JSON_PATH = ML_OUTPUT_DIR / "model_analysis.json"


def load_experiments():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing experiment registry: {REGISTRY_PATH}")

    return pd.read_csv(REGISTRY_PATH)


def add_stability_metrics(df):
    df = df.copy()

    df["cv_accuracy_cv_ratio"] = (
        df["cv_accuracy_std"] / df["cv_accuracy_mean"]
    )

    df["cv_log_loss_cv_ratio"] = (
        df["cv_log_loss_std"] / df["cv_log_loss_mean"]
    )

    return df


def summarize_by_feature_set(df):
    return (
        df.groupby("feature_set")
        .agg(
            mean_test_accuracy=("accuracy", "mean"),
            mean_cv_accuracy=("cv_accuracy_mean", "mean"),
            mean_test_log_loss=("log_loss", "mean"),
            mean_cv_log_loss=("cv_log_loss_mean", "mean"),
            mean_cv_accuracy_std=("cv_accuracy_std", "mean"),
            mean_cv_log_loss_std=("cv_log_loss_std", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_train_seconds=("train_seconds", "mean"),
            experiment_count=("model", "count"),
        )
        .reset_index()
        .sort_values("mean_cv_log_loss")
    )


def summarize_by_model(df):
    return (
        df.groupby("model")
        .agg(
            mean_test_accuracy=("accuracy", "mean"),
            mean_cv_accuracy=("cv_accuracy_mean", "mean"),
            mean_test_log_loss=("log_loss", "mean"),
            mean_cv_log_loss=("cv_log_loss_mean", "mean"),
            mean_cv_accuracy_std=("cv_accuracy_std", "mean"),
            mean_cv_log_loss_std=("cv_log_loss_std", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_train_seconds=("train_seconds", "mean"),
            experiment_count=("feature_set", "count"),
        )
        .reset_index()
        .sort_values("mean_cv_log_loss")
    )


def get_leaders(df):
    return {
        "best_test_accuracy": df.sort_values(
            "accuracy",
            ascending=False,
        ).iloc[0].to_dict(),
        "best_cv_accuracy": df.sort_values(
            "cv_accuracy_mean",
            ascending=False,
        ).iloc[0].to_dict(),
        "best_test_log_loss": df.sort_values("log_loss").iloc[0].to_dict(),
        "best_cv_log_loss": df.sort_values("cv_log_loss_mean").iloc[0].to_dict(),
        "most_stable_cv_accuracy": df.sort_values(
            "cv_accuracy_cv_ratio"
        ).iloc[0].to_dict(),
        "most_stable_cv_log_loss": df.sort_values(
            "cv_log_loss_cv_ratio"
        ).iloc[0].to_dict(),
        "best_brier_score": df.sort_values("brier_score").iloc[0].to_dict(),
    }


def build_recommendation(leaders):
    best_cv_log_loss = leaders["best_cv_log_loss"]
    best_cv_accuracy = leaders["best_cv_accuracy"]
    best_test_accuracy = leaders["best_test_accuracy"]

    recommendation = (
        f"The strongest probability-quality baseline is "
        f"{best_cv_log_loss['model']} with the "
        f"{best_cv_log_loss['feature_set']} feature set, based on the lowest "
        f"mean cross-validation log loss "
        f"({best_cv_log_loss['cv_log_loss_mean']:.4f}). "
        f"The strongest cross-validation accuracy baseline is "
        f"{best_cv_accuracy['model']} with the "
        f"{best_cv_accuracy['feature_set']} feature set "
        f"({best_cv_accuracy['cv_accuracy_mean']:.4f}). "
        f"The strongest hold-out accuracy result is "
        f"{best_test_accuracy['model']} with the "
        f"{best_test_accuracy['feature_set']} feature set "
        f"({best_test_accuracy['accuracy']:.4f}). "
        f"For the World Cup simulator, probability quality should be prioritized "
        f"over raw accuracy, so models with strong log loss should be favored "
        f"for tuning and calibration."
    )

    return recommendation


def write_markdown_report(
    experiment_df,
    feature_summary,
    model_summary,
    leaders,
    recommendation,
):
    top_experiments = experiment_df[
        [
            "model",
            "feature_set",
            "accuracy",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "log_loss",
            "cv_log_loss_mean",
            "cv_log_loss_std",
            "brier_score",
            "train_seconds",
        ]
    ].sort_values("cv_log_loss_mean").head(10)

    markdown = f"""# World Cup 2026 Model Analysis

## Recommendation

{recommendation}

## Leaders

| Category | Model | Feature Set | Value |
|---|---|---|---:|
| Best Test Accuracy | {leaders['best_test_accuracy']['model']} | {leaders['best_test_accuracy']['feature_set']} | {leaders['best_test_accuracy']['accuracy']:.4f} |
| Best CV Accuracy | {leaders['best_cv_accuracy']['model']} | {leaders['best_cv_accuracy']['feature_set']} | {leaders['best_cv_accuracy']['cv_accuracy_mean']:.4f} |
| Best Test Log Loss | {leaders['best_test_log_loss']['model']} | {leaders['best_test_log_loss']['feature_set']} | {leaders['best_test_log_loss']['log_loss']:.4f} |
| Best CV Log Loss | {leaders['best_cv_log_loss']['model']} | {leaders['best_cv_log_loss']['feature_set']} | {leaders['best_cv_log_loss']['cv_log_loss_mean']:.4f} |
| Best Brier Score | {leaders['best_brier_score']['model']} | {leaders['best_brier_score']['feature_set']} | {leaders['best_brier_score']['brier_score']:.4f} |

## Top Experiments by CV Log Loss

{top_experiments.to_markdown(index=False)}

## Feature Set Summary

{feature_summary.to_markdown(index=False)}

## Model Family Summary

{model_summary.to_markdown(index=False)}

## Interpretation Notes

- CV log loss is especially important because the simulator depends on probability estimates.
- Accuracy is useful, but it should not dominate model selection.
- Lower CV standard deviation suggests more stable performance across folds.
- Simpler feature sets that perform competitively should remain candidates for production.
"""

    ANALYSIS_MD_PATH.write_text(markdown, encoding="utf-8")


def save_json(leaders, recommendation, feature_summary, model_summary):
    payload = {
        "leaders": leaders,
        "recommendation": recommendation,
        "feature_summary": feature_summary.to_dict(orient="records"),
        "model_summary": model_summary.to_dict(orient="records"),
    }

    with open(ANALYSIS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    experiment_df = load_experiments()
    experiment_df = add_stability_metrics(experiment_df)

    feature_summary = summarize_by_feature_set(experiment_df)
    model_summary = summarize_by_model(experiment_df)
    leaders = get_leaders(experiment_df)
    recommendation = build_recommendation(leaders)

    experiment_df.to_csv(ANALYSIS_CSV_PATH, index=False)

    write_markdown_report(
        experiment_df=experiment_df,
        feature_summary=feature_summary,
        model_summary=model_summary,
        leaders=leaders,
        recommendation=recommendation,
    )

    save_json(
        leaders=leaders,
        recommendation=recommendation,
        feature_summary=feature_summary,
        model_summary=model_summary,
    )

    print("Model analysis complete.")
    print(f"CSV: {ANALYSIS_CSV_PATH}")
    print(f"Markdown: {ANALYSIS_MD_PATH}")
    print(f"JSON: {ANALYSIS_JSON_PATH}")
    print()
    print("Recommendation:")
    print(recommendation)


if __name__ == "__main__":
    main()