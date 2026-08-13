#compare_simulation_to_history.py

from pathlib import Path
import pandas as pd


HISTORICAL_FILE = Path("outputs/historical_world_cups/historical_world_cup_summary.csv")
SIMULATION_FILE = Path("outputs/monte_carlo/simulation_statistics.csv")
OUTPUT_DIR = Path("outputs/model_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    historical = pd.read_csv(HISTORICAL_FILE)
    simulation = pd.read_csv(SIMULATION_FILE)

    hist_avg = historical[historical["year"].astype(str) == "average"].iloc[0]
    sim = simulation.iloc[0]

    rows = [
        {
            "metric": "goals_per_match",
            "historical": hist_avg["goals_per_match"],
            "simulation": sim["avg_goals_per_match"],
        },
        {
            "metric": "zero_zero_rate",
            "historical": hist_avg["zero_zero_rate"],
            "simulation": None,
        },
        {
            "metric": "three_plus_goal_rate",
            "historical": hist_avg["three_plus_goal_rate"],
            "simulation": None,
        },
        {
            "metric": "avg_margin",
            "historical": hist_avg["avg_margin"],
            "simulation": None,
        },
        {
            "metric": "knockout_extra_time_rate",
            "historical": None,
            "simulation": sim["extra_time_rate_knockout_matches"],
        },
        {
            "metric": "knockout_penalty_rate",
            "historical": None,
            "simulation": sim["penalty_rate_knockout_matches"],
        },
    ]

    comparison = pd.DataFrame(rows)
    comparison["difference"] = comparison["simulation"] - comparison["historical"]

    out_file = OUTPUT_DIR / "historical_vs_simulation.csv"
    comparison.to_csv(out_file, index=False)

    print("Historical vs Simulation")
    print("------------------------")
    print(comparison.to_string(index=False))

    print()
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()