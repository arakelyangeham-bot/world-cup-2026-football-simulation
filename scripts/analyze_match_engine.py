#analyze_match_engine.py

from collections import Counter
from pathlib import Path
import random

import numpy as np
import pandas as pd
import argparse

from match_engine import poisson_expected_goals, simulate_poisson_score
from team_strength_loader import load_poisson_team_strengths
from simulation_utils import canonical_scoreline


OUTPUT_DIR = Path("outputs/match_engine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_matchup(team_a: str, team_b: str, n: int = 10000, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    strengths = load_poisson_team_strengths()

    if team_a not in strengths:
        raise ValueError(f"Missing team strength for {team_a}")

    if team_b not in strengths:
        raise ValueError(f"Missing team strength for {team_b}")

    team_a_strength = strengths[team_a]
    team_b_strength = strengths[team_b]

    lambda_a, lambda_b = poisson_expected_goals(team_a_strength, team_b_strength)

    a_wins = 0
    draws = 0
    b_wins = 0
    scorelines = Counter()

    total_goals_a = 0
    total_goals_b = 0

    for _ in range(n):
        goals_a, goals_b = simulate_poisson_score(team_a_strength, team_b_strength)

        total_goals_a += goals_a
        total_goals_b += goals_b

        scorelines[canonical_scoreline(goals_a, goals_b)] += 1

        if goals_a > goals_b:
            a_wins += 1
        elif goals_b > goals_a:
            b_wins += 1
        else:
            draws += 1

    scoreline_rows = [
        {
            "scoreline": scoreline,
            "count": count,
            "probability": count / n,
        }
        for scoreline, count in scorelines.most_common()
    ]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "simulations": n,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "avg_goals_a": total_goals_a / n,
        "avg_goals_b": total_goals_b / n,
        "team_a_win_prob": a_wins / n,
        "draw_prob": draws / n,
        "team_b_win_prob": b_wins / n,
        "scorelines": scoreline_rows,
    }


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def write_summary_csv(result: dict) -> None:
    summary_file = (
        OUTPUT_DIR
        / f"{safe_filename(result['team_a'])}_vs_{safe_filename(result['team_b'])}_summary.csv"
    )

    row = {
        "team_a": result["team_a"],
        "team_b": result["team_b"],
        "simulations": result["simulations"],
        "lambda_a": result["lambda_a"],
        "lambda_b": result["lambda_b"],
        "avg_goals_a": result["avg_goals_a"],
        "avg_goals_b": result["avg_goals_b"],
        "team_a_win_prob": result["team_a_win_prob"],
        "draw_prob": result["draw_prob"],
        "team_b_win_prob": result["team_b_win_prob"],
    }

    pd.DataFrame([row]).to_csv(summary_file, index=False)
    print(f"Summary saved -> {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("team_a", nargs="?", default="Spain")
    parser.add_argument("team_b", nargs="?", default="Argentina")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    result = simulate_matchup(
        args.team_a,
        args.team_b,
        n=args.n,
        seed=args.seed,
    )

    print("Match Engine Diagnostic")
    print("-----------------------")
    print(f"{result['team_a']} vs {result['team_b']}")
    print(f"Simulations: {result['simulations']}")

    print()
    print("Expected goals")
    print(f"{result['team_a']}: {result['lambda_a']:.3f}")
    print(f"{result['team_b']}: {result['lambda_b']:.3f}")

    print()
    print("Outcome probabilities")
    print(f"{result['team_a']} win: {result['team_a_win_prob']:.3f}")
    print(f"Draw: {result['draw_prob']:.3f}")
    print(f"{result['team_b']} win: {result['team_b_win_prob']:.3f}")

    print()
    print("Top scorelines")
    scoreline_df = pd.DataFrame(result["scorelines"])
    print(scoreline_df.head(15).to_string(index=False))

    scoreline_file = (
        OUTPUT_DIR
        / f"{safe_filename(result['team_a'])}_vs_{safe_filename(result['team_b'])}_scorelines.csv"
    )
    scoreline_df.to_csv(scoreline_file, index=False)

    print()
    print(f"Scorelines saved -> {scoreline_file}")
    write_summary_csv(result)


if __name__ == "__main__":
    main()