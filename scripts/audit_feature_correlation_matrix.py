#audit_feature_correlation_matrix.py

from pathlib import Path

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


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    available = [feature for feature in FEATURES if feature in df.columns]

    corr = df[available].corr()

    print(f"Rows: {len(df)}")
    print(f"Features: {len(available)}")

    print()
    print("Selected feature correlations")
    print("-----------------------------")

    pairs = [
        ("attack_diff", "fifa_points_diff"),
        ("poisson_attack_diff", "fifa_points_diff"),
        ("attack_diff", "poisson_attack_diff"),
        ("defense_diff", "poisson_defense_diff"),
        ("home_attack", "home_poisson_attack"),
        ("away_attack", "away_poisson_attack"),
        ("home_defense", "home_poisson_defense"),
        ("away_defense", "away_poisson_defense"),
        ("home_fifa_points", "home_attack"),
        ("away_fifa_points", "away_attack"),
    ]

    for left, right in pairs:
        if left in corr.index and right in corr.columns:
            print(
                f"{left:<28} vs {right:<28} "
                f"{corr.loc[left, right]:>7.3f}"
            )

    print()
    print("Highly correlated feature pairs |r| >= 0.70")
    print("------------------------------------------")

    seen = set()

    rows = []

    for left in available:
        for right in available:
            if left == right:
                continue

            key = tuple(sorted([left, right]))
            if key in seen:
                continue

            seen.add(key)

            value = corr.loc[left, right]

            if abs(value) >= 0.70:
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": value,
                    }
                )

    rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)

    if not rows:
        print("None")

    for row in rows:
        print(
            f"{row['left']:<28} "
            f"{row['right']:<28} "
            f"{row['correlation']:>7.3f}"
        )

    print()
    print("Full correlation matrix")
    print("-----------------------")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()