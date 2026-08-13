#results_summary.py

import pandas as pd


def build_summary_row(
    model_name,
    feature_set_name,
    metrics,
    metadata,
):
    """
    Build one standardized experiment-summary row.
    """

    row = {
        "model": model_name,
        "feature_set": feature_set_name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "log_loss": metrics.get("log_loss"),
        "brier_score": metrics.get("brier_score"),
        "cv_accuracy_mean": metadata.get("cv_accuracy_mean"),
        "cv_accuracy_std": metadata.get("cv_accuracy_std"),

        "cv_balanced_accuracy_mean": metadata.get(
            "cv_balanced_accuracy_mean"
        ),
        "cv_balanced_accuracy_std": metadata.get(
            "cv_balanced_accuracy_std"
        ),

        "cv_log_loss_mean": metadata.get("cv_log_loss_mean"),
        "cv_log_loss_std": metadata.get("cv_log_loss_std"),
        "rows_used": metadata.get("rows_used"),
        "train_rows": metadata.get("train_rows"),
        "test_rows": metadata.get("test_rows"),
        "n_features": len(metadata.get("features", [])),
        "train_seconds": metadata.get("train_seconds"),
        "random_state": metadata.get("random_state"),
        "test_size": metadata.get("test_size"),
    }

    return row


def save_summary(rows, output_path):
    """
    Save a list of experiment rows to CSV.
    """

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        by=["model", "log_loss", "balanced_accuracy"],
        ascending=[True, True, False],
    )

    summary_df.to_csv(output_path, index=False)

    return summary_df

def print_summary_table(summary_df):
    """
    Print a compact, readable model-summary table.
    """

    print("=" * 100)
    print("MODEL SUMMARY")
    print("=" * 100)
    print(
        f"{'Model':20s} "
        f"{'Feature Set':14s} "
        f"{'Test Acc':>9s} "
        f"{'CV Acc':>16s} "
        f"{'Test LogLoss':>13s} "
        f"{'CV LogLoss':>18s} "
        f"{'Feat':>5s}"
    )
    print("-" * 100)

    for _, row in summary_df.iterrows():
        cv_acc = f"{row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}"
        cv_log_loss = f"{row['cv_log_loss_mean']:.4f} ± {row['cv_log_loss_std']:.4f}"

        print(
            f"{row['model']:20s} "
            f"{row['feature_set']:14s} "
            f"{row['accuracy']:9.4f} "
            f"{cv_acc:>16s} "
            f"{row['log_loss']:13.4f} "
            f"{cv_log_loss:>18s} "
            f"{int(row['n_features']):5d}"
        )

    print("=" * 100)