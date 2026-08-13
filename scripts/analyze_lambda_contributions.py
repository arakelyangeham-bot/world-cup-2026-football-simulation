#analyze_lambda_contributions.py

from itertools import combinations
from pathlib import Path

import pandas as pd

from team_strength_loader import load_poisson_team_strengths


BASE_GOALS = 1.35
MEAN_ATTACK = 0.295
MEAN_DEFENSE = 0.636
ATTACK_WEIGHT = 1.2
DEFENSE_WEIGHT = 0.8
MIN_LAMBDA = 0.2

OUTPUT_DIR = Path("outputs/match_engine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def contribution_rows(team_a, team_b, strengths):
    rows = []

    for attacking_team, defending_team in [(team_a, team_b), (team_b, team_a)]:
        attack_deviation = strengths[attacking_team]["attack"] - MEAN_ATTACK
        opponent_defense_adjustment = MEAN_DEFENSE - strengths[defending_team]["defense"]

        attack_contribution = ATTACK_WEIGHT * attack_deviation
        defense_contribution = -DEFENSE_WEIGHT * opponent_defense_adjustment

        raw_lambda = BASE_GOALS + attack_contribution + defense_contribution
        final_lambda = max(MIN_LAMBDA, raw_lambda)

        rows.append(
            {
                "attacking_team": attacking_team,
                "defending_team": defending_team,
                "attack_value": strengths[attacking_team]["attack"],
                "defense_value_opponent": strengths[defending_team]["defense"],
                "attack_deviation": attack_deviation,
                "opponent_defense_adjustment": opponent_defense_adjustment,
                "base_contribution": BASE_GOALS,
                "attack_contribution": attack_contribution,
                "defense_contribution": defense_contribution,
                "raw_lambda": raw_lambda,
                "final_lambda": final_lambda,
                "lambda_was_clipped": raw_lambda < MIN_LAMBDA,
            }
        )

    return rows


def main():
    strengths = load_poisson_team_strengths()
    teams = sorted(strengths)

    rows = []

    for team_a, team_b in combinations(teams, 2):
        rows.extend(contribution_rows(team_a, team_b, strengths))

    df = pd.DataFrame(rows)

    out_file = OUTPUT_DIR / "lambda_contribution_analysis.csv"
    df.to_csv(out_file, index=False)

    summary = df[
        [
            "attack_contribution",
            "defense_contribution",
            "raw_lambda",
            "final_lambda",
        ]
    ].describe()

    summary_file = OUTPUT_DIR / "lambda_contribution_summary.csv"
    summary.to_csv(summary_file)

    print("Lambda Contribution Analysis")
    print("----------------------------")
    print(f"Teams: {len(teams)}")
    print(f"Unique matchups: {len(list(combinations(teams, 2)))}")
    print(f"Directional rows: {len(df)}")

    print()
    print(summary.to_string())

    print()
    print("Clipped lambda rows:", int(df["lambda_was_clipped"].sum()))
    print(f"Clipped lambda rate: {df['lambda_was_clipped'].mean():.3f}")

    print()
    print("Largest attack contributions")
    print(df.sort_values("attack_contribution", ascending=False).head(10)[
        ["attacking_team", "defending_team", "attack_contribution", "defense_contribution", "final_lambda"]
    ].to_string(index=False))

    print()
    print("Largest defensive suppressions")
    print(df.sort_values("defense_contribution", ascending=True).head(10)[
        ["attacking_team", "defending_team", "attack_contribution", "defense_contribution", "final_lambda"]
    ].to_string(index=False))

    print()
    print(f"Saved -> {out_file}")
    print(f"Saved -> {summary_file}")


if __name__ == "__main__":
    main()