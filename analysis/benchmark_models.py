# benchmark_models.py

from pathlib import Path

import pandas as pd

from shared.config import ML_OUTPUT_DIR


REGISTRY_PATH = ML_OUTPUT_DIR / "experiment_results.csv"
REPORT_CSV_PATH = ML_OUTPUT_DIR / "benchmark_report.csv"
REPORT_MD_PATH = ML_OUTPUT_DIR / "benchmark_report.md"


def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")

    return pd.read_csv(REGISTRY_PATH)


def build_benchmark_report(df):
    report_df = df.copy()

    report_df["rank_cv_log_loss"] = report_df["cv_log_loss_mean"].rank(
        method="min",
        ascending=True,
    )

    report_df["rank_cv_accuracy"] = report_df["cv_accuracy_mean"].rank(
        method="min",
        ascending=False,
    )

    report_df["rank_test_log_loss"] = report_df["log_loss"].rank(
        method="min",
        ascending=True,
    )

    report_df["rank_brier_score"] = report_df["brier_score"].rank(
        method="min",
        ascending=True,
    )

    report_df["rank_test_accuracy"] = report_df["accuracy"].rank(
        method="min",
        ascending=False,
    )

    report_df["average_rank"] = report_df[
        [
            "rank_cv_log_loss",
            "rank_brier_score",
            "rank_cv_accuracy",
            "rank_test_log_loss",
            "rank_test_accuracy",
        ]
    ].mean(axis=1)

    report_df = report_df.sort_values(
        by=["average_rank", "cv_log_loss_mean"],
        ascending=[True, True],
    )

    return report_df


def write_markdown_report(report_df):
    best_cv_log_loss = report_df.sort_values("cv_log_loss_mean").iloc[0]
    best_cv_accuracy = report_df.sort_values(
        "cv_accuracy_mean",
        ascending=False,
    ).iloc[0]
    best_test_log_loss = report_df.sort_values("log_loss").iloc[0]
    best_test_accuracy = report_df.sort_values(
        "accuracy",
        ascending=False,
    ).iloc[0]

    top_table = report_df[
        [
            "model",
            "feature_set",
            "accuracy",
            "cv_accuracy_mean",
            "log_loss",
            "cv_log_loss_mean",
            "brier_score",
            "average_rank",
        ]
    ].head(10)

    markdown = f"""# World Cup 2026 ML Benchmark Report

## Current Leaders

| Metric | Model | Feature Set | Value |
|---|---|---|---:|
| Best CV Log Loss | {best_cv_log_loss['model']} | {best_cv_log_loss['feature_set']} | {best_cv_log_loss['cv_log_loss_mean']:.4f} |
| Best CV Accuracy | {best_cv_accuracy['model']} | {best_cv_accuracy['feature_set']} | {best_cv_accuracy['cv_accuracy_mean']:.4f} |
| Best Test Log Loss | {best_test_log_loss['model']} | {best_test_log_loss['feature_set']} | {best_test_log_loss['log_loss']:.4f} |
| Best Test Accuracy | {best_test_accuracy['model']} | {best_test_accuracy['feature_set']} | {best_test_accuracy['accuracy']:.4f} |

## Top Experiments by Average Rank

{top_table.to_markdown(index=False)}

## Notes

- CV metrics are more stable than the single hold-out test split.
- Log loss is especially important because the simulator needs useful probabilities, not just class predictions.
- Accuracy is still useful, but it should not be the only model-selection criterion.
"""

    REPORT_MD_PATH.write_text(markdown, encoding="utf-8")


def main():
    df = load_registry()
    report_df = build_benchmark_report(df)

    report_df.to_csv(REPORT_CSV_PATH, index=False)
    write_markdown_report(report_df)

    print("Benchmark report created.")
    print(f"CSV: {REPORT_CSV_PATH}")
    print(f"Markdown: {REPORT_MD_PATH}")

    print()
    print("Top 10 experiments:")
    print(
        report_df[
            [
                "model",
                "feature_set",
                "accuracy",
                "cv_accuracy_mean",
                "log_loss",
                "cv_log_loss_mean",
                "average_rank",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()