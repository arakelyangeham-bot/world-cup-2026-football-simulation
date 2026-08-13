#feature_distribution_audit.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.team_strength_loader import load_team_repository


HISTORICAL_DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "feature_distribution_summary.csv"
QUANTILES_OUTPUT_PATH = OUTPUT_DIR / "feature_distribution_quantiles.csv"


FEATURES = [
    "poisson_attack",
    "poisson_defense",
    "fifa_points",
]


def historical_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for side in ["home", "away"]:
        for _, row in df.iterrows():
            rows.append(
                {
                    "source": "historical_training",
                    "team": row[f"{side}_team"],
                    "poisson_attack": row[f"{side}_poisson_attack"],
                    "poisson_defense": row[f"{side}_poisson_defense"],
                    "fifa_points": row[f"{side}_fifa_points"],
                }
            )

    return pd.DataFrame(rows)


def runtime_feature_frame(team_repository: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for team, values in team_repository.items():
        rows.append(
            {
                "source": "runtime_repository",
                "team": team,
                "poisson_attack": values["poisson_attack"],
                "poisson_defense": values["poisson_defense"],
                "fifa_points": values["fifa_points"],
            }
        )

    return pd.DataFrame(rows)


def summarize_feature(
    df: pd.DataFrame,
    source: str,
    feature: str,
) -> dict:
    values = pd.to_numeric(
        df.loc[df["source"] == source, feature],
        errors="coerce",
    ).dropna()

    return {
        "source": source,
        "feature": feature,
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p25": values.quantile(0.25),
        "median": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "max": values.max(),
    }


def main() -> None:
    historical_df = pd.read_csv(HISTORICAL_DATASET_PATH)
    team_repository = load_team_repository()

    features_df = pd.concat(
        [
            historical_feature_frame(historical_df),
            runtime_feature_frame(team_repository),
        ],
        ignore_index=True,
    )

    summary_rows = []

    for feature in FEATURES:
        historical_summary = summarize_feature(
            features_df,
            "historical_training",
            feature,
        )
        runtime_summary = summarize_feature(
            features_df,
            "runtime_repository",
            feature,
        )

        comparison = {
            "feature": feature,
            "historical_count": historical_summary["count"],
            "runtime_count": runtime_summary["count"],
            "historical_mean": historical_summary["mean"],
            "runtime_mean": runtime_summary["mean"],
            "mean_difference_runtime_minus_historical": (
                runtime_summary["mean"] - historical_summary["mean"]
            ),
            "mean_ratio_runtime_to_historical": (
                runtime_summary["mean"] / historical_summary["mean"]
                if historical_summary["mean"] != 0
                else None
            ),
            "historical_std": historical_summary["std"],
            "runtime_std": runtime_summary["std"],
            "historical_min": historical_summary["min"],
            "runtime_min": runtime_summary["min"],
            "historical_p25": historical_summary["p25"],
            "runtime_p25": runtime_summary["p25"],
            "historical_median": historical_summary["median"],
            "runtime_median": runtime_summary["median"],
            "historical_p75": historical_summary["p75"],
            "runtime_p75": runtime_summary["p75"],
            "historical_max": historical_summary["max"],
            "runtime_max": runtime_summary["max"],
        }

        summary_rows.append(comparison)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    quantile_rows = []

    for source in ["historical_training", "runtime_repository"]:
        for feature in FEATURES:
            values = pd.to_numeric(
                features_df.loc[features_df["source"] == source, feature],
                errors="coerce",
            ).dropna()

            for q in [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]:
                quantile_rows.append(
                    {
                        "source": source,
                        "feature": feature,
                        "quantile": q,
                        "value": values.quantile(q),
                    }
                )

    quantiles = pd.DataFrame(quantile_rows)
    quantiles.to_csv(QUANTILES_OUTPUT_PATH, index=False)

    print("Feature Distribution Audit")
    print("--------------------------")
    print(summary.round(6).to_string(index=False))
    print()
    print(f"Wrote summary   -> {SUMMARY_OUTPUT_PATH}")
    print(f"Wrote quantiles -> {QUANTILES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()