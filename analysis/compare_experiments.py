import pandas as pd

from shared.config import ML_OUTPUT_DIR


REGISTRY_PATH = ML_OUTPUT_DIR / "experiment_results.csv"
OUTPUT_DIR = ML_OUTPUT_DIR / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


COLUMN_ALIASES = {
    "model": ["Model", "model"],
    "feature_set": ["Feature Set", "feature_set"],
    "test_acc": ["Test Acc", "test_accuracy", "accuracy"],
    "test_logloss": ["Test LogLoss", "test_log_loss", "log_loss"],
    "cv_logloss": ["CV LogLoss", "cv_log_loss_mean", "cv_log_loss"],
}


def find_column(df, logical_name):
    for candidate in COLUMN_ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"Could not find column for {logical_name}. "
        f"Available columns: {df.columns.tolist()}"
    )


def main():
    df = pd.read_csv(REGISTRY_PATH)

    df = df[df["feature_set"].isin(["v1", "v2"])].copy()

    comparison_rows = []

    for model in sorted(df["model"].unique()):

        model_df = df[df["model"] == model]

        if not {"v1", "v2"}.issubset(set(model_df["feature_set"])):
            continue

        v1 = model_df[model_df["feature_set"] == "v1"].iloc[-1]
        v2 = model_df[model_df["feature_set"] == "v2"].iloc[-1]

        comparison_rows.append(
            {
                "Model": model,

                "V1 Accuracy": v1["accuracy"],
                "V2 Accuracy": v2["accuracy"],
                "Δ Accuracy": v2["accuracy"] - v1["accuracy"],

                "V1 Log Loss": v1["log_loss"],
                "V2 Log Loss": v2["log_loss"],
                "Δ Log Loss": v2["log_loss"] - v1["log_loss"],

                "V1 CV Log Loss": v1["cv_log_loss_mean"],
                "V2 CV Log Loss": v2["cv_log_loss_mean"],
                "Δ CV Log Loss": (
                    v2["cv_log_loss_mean"]
                    - v1["cv_log_loss_mean"]
                ),

                "V1 Brier": v1["brier_score"],
                "V2 Brier": v2["brier_score"],
                "Δ Brier": (
                    v2["brier_score"]
                    - v1["brier_score"]
                ),
            }
        )

    results = pd.DataFrame(comparison_rows)

    csv_path = OUTPUT_DIR / "feature_set_comparison.csv"
    results.to_csv(csv_path, index=False)

    print()
    print("=" * 80)
    print("VERSION COMPARISON")
    print("=" * 80)
    print(results.to_string(index=False))
    print()
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()