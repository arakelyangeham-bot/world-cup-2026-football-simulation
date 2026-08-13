# feature_importance.py

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


def save_feature_importance(model, feature_names, output_prefix):
    """
    Save feature importance CSV and PNG for models that expose
    feature_importances_.
    """

    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature_importances_; skipping.")
        return None

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(0, "rank", importance_df.index + 1)

    csv_path = f"{output_prefix}_feature_importance.csv"
    png_path = f"{output_prefix}_feature_importance.png"

    importance_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved feature importance CSV: {csv_path}")
    print(f"Saved feature importance plot: {png_path}")

    return importance_df