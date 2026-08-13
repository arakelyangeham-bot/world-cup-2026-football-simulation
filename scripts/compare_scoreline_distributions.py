#compare_scoreline_distributions.py

from pathlib import Path
import pandas as pd


HISTORICAL_FILE = Path("outputs/historical_world_cups/historical_scoreline_distribution.csv")
SIMULATION_FILE = Path("outputs/monte_carlo/simulation_scoreline_distribution.csv")
OUTPUT_DIR = Path("outputs/model_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    historical = pd.read_csv(HISTORICAL_FILE).rename(
        columns={
            "count": "historical_count",
            "probability": "historical_probability",
        }
    )

    simulation = pd.read_csv(SIMULATION_FILE).rename(
        columns={
            "count": "simulation_count",
            "probability": "simulation_probability",
        }
    )

    comparison = historical.merge(
        simulation,
        on="scoreline",
        how="outer",
    )

    comparison["historical_count"] = comparison["historical_count"].fillna(0).astype(int)
    comparison["simulation_count"] = comparison["simulation_count"].fillna(0).astype(int)
    comparison["historical_probability"] = comparison["historical_probability"].fillna(0.0)
    comparison["simulation_probability"] = comparison["simulation_probability"].fillna(0.0)

    comparison["difference"] = (
        comparison["simulation_probability"]
        - comparison["historical_probability"]
    )
    comparison["absolute_difference"] = comparison["difference"].abs()

    comparison = comparison.sort_values(
        "absolute_difference",
        ascending=False,
    ).reset_index(drop=True)

    total_variation_distance = 0.5 * comparison["absolute_difference"].sum()

    out_file = OUTPUT_DIR / "scoreline_distribution_comparison.csv"
    comparison.to_csv(out_file, index=False)

    print("Scoreline Distribution Comparison")
    print("---------------------------------")
    print(comparison.head(25).to_string(index=False))

    print()
    print(f"Total variation distance: {total_variation_distance:.6f}")
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()