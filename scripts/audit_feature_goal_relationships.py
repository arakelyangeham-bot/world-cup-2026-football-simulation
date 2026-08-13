#audit_feature_goal_relationships.py

from pathlib import Path
from statistics import mean

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


FEATURES = [
    "home_attack",
    "away_attack",
    "home_defense",
    "away_defense",
    "home_gk",
    "away_gk",
    "home_poisson_attack",
    "away_poisson_attack",
    "home_poisson_defense",
    "away_poisson_defense",
    "attack_diff",
    "defense_diff",
    "gk_diff",
    "poisson_attack_diff",
    "poisson_defense_diff",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
]


TARGETS = [
    "home_score",
    "away_score",
    "total_goals",
    "goal_diff",
]


def corr(df: pd.DataFrame, x: str, y: str) -> float:
    subset = df[[x, y]].dropna()

    if len(subset) < 2:
        return float("nan")

    return subset[x].corr(subset[y])


def print_top_correlations(df: pd.DataFrame, target: str) -> None:
    rows = []

    for feature in FEATURES:
        if feature not in df.columns:
            continue

        value = corr(df, feature, target)

        rows.append(
            {
                "feature": feature,
                "correlation": value,
                "abs_correlation": abs(value),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: r["abs_correlation"],
        reverse=True,
    )

    print()
    print(f"Top correlations with {target}")
    print("-" * (22 + len(target)))

    for row in rows[:10]:
        print(
            f"{row['feature']:<30} "
            f"{row['correlation']:>8.3f}"
        )


def print_score_summary(df: pd.DataFrame) -> None:
    print()
    print("Score summary")
    print("-------------")
    print(f"Matches: {len(df)}")
    print(f"Avg home goals: {df['home_score'].mean():.3f}")
    print(f"Avg away goals: {df['away_score'].mean():.3f}")
    print(f"Avg total goals: {df['total_goals'].mean():.3f}")
    print(f"Avg goal diff: {df['goal_diff'].mean():.3f}")


def print_result_summary(df: pd.DataFrame) -> None:
    print()
    print("Result summary")
    print("--------------")
    counts = df["result"].value_counts()
    rates = df["result"].value_counts(normalize=True)

    for result, count in counts.items():
        print(f"{result:<10} {count:>5}  {rates[result]:.3f}")


def print_feature_means_by_result(df: pd.DataFrame) -> None:
    print()
    print("Feature means by result")
    print("-----------------------")

    selected = [
        "attack_diff",
        "defense_diff",
        "gk_diff",
        "poisson_attack_diff",
        "poisson_defense_diff",
        "fifa_points_diff",
    ]

    grouped = df.groupby("result")[selected].mean()

    print(grouped.round(3).to_string())


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print_score_summary(df)
    print_result_summary(df)

    for target in TARGETS:
        print_top_correlations(df, target)

    print_feature_means_by_result(df)


if __name__ == "__main__":
    main()