# shared/experiment_registry.py

import pandas as pd


def register_experiments(summary_df, registry_path):
    """
    Append experiment results to the global experiment registry.

    Existing rows for the same model + feature_set are replaced,
    so rerunning a model updates its latest results instead of duplicating rows.
    """

    if registry_path.exists():
        registry_df = pd.read_csv(registry_path)
    else:
        registry_df = pd.DataFrame()

    combined_df = pd.concat(
        [registry_df, summary_df],
        ignore_index=True,
    )

    combined_df = combined_df.drop_duplicates(
        subset=["model", "feature_set"],
        keep="last",
    )

    combined_df = combined_df.sort_values(
        by=["cv_log_loss_mean", "cv_accuracy_mean"],
        ascending=[True, False],
    )

    combined_df.to_csv(registry_path, index=False)

    return combined_df