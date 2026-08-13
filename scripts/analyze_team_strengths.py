#analyze_team_strengths.py

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAM_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "team_strength_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize(series: pd.Series) -> dict:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def print_summary(title: str, series: pd.Series):
    stats = summarize(series)

    print()
    print(title)
    print("-" * len(title))

    for key, value in stats.items():
        print(f"{key:<8}: {value:.3f}" if isinstance(value, float) else f"{key:<8}: {value}")


def main():
    df = pd.read_csv(TEAM_FILE)

    attack = df["poisson_attack_adj"]
    defense = df["poisson_defense_adj"]

    print("=" * 60)
    print("Poisson Team Strength Analysis")
    print("=" * 60)

    print_summary("Attack Adjustment", attack)
    print_summary("Defense Adjustment", defense)

    print()
    print("=" * 60)
    print("Top 10 Attack")
    print("=" * 60)

    top_attack = (
        df[["nation", "poisson_attack_adj"]]
        .sort_values("poisson_attack_adj", ascending=False)
    )

    print(top_attack.head(10).to_string(index=False))

    print()
    print("=" * 60)
    print("Bottom 10 Attack")
    print("=" * 60)

    print(top_attack.tail(10).to_string(index=False))

    print()
    print("=" * 60)
    print("Best 10 Defenses")
    print("=" * 60)

    # Lower defense values are better
    best_defense = (
        df[["nation", "poisson_defense_adj"]]
        .sort_values("poisson_defense_adj", ascending=True)
    )

    print(best_defense.head(10).to_string(index=False))

    print()
    print("=" * 60)
    print("Worst 10 Defenses")
    print("=" * 60)

    print(best_defense.tail(10).to_string(index=False))

    # Save ranked tables
    top_attack.to_csv(
        OUTPUT_DIR / "attack_rankings.csv",
        index=False,
    )

    best_defense.to_csv(
        OUTPUT_DIR / "defense_rankings.csv",
        index=False,
    )

    # Correlation
    print()
    print("=" * 60)
    print("Attack vs Defense Correlation")
    print("=" * 60)

    corr = attack.corr(defense)
    print(f"Pearson correlation: {corr:.3f}")

    summary = pd.DataFrame(
        [
            {
                "attack_mean": attack.mean(),
                "attack_std": attack.std(),
                "attack_min": attack.min(),
                "attack_max": attack.max(),
                "defense_mean": defense.mean(),
                "defense_std": defense.std(),
                "defense_min": defense.min(),
                "defense_max": defense.max(),
                "attack_defense_correlation": corr,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "strength_summary.csv",
        index=False,
    )

    print()
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()