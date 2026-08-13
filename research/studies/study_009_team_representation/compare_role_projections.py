#compare_role_projections.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.roster_builder import RosterBuilder
from research.player_intelligence.role_projection import weighted_role_score


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "study_009_team_representation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "role_projection_comparison.csv"


PROJECTIONS = {
    "current": {
        "attack": {"ST": 0.40, "W": 0.25, "AM": 0.20, "CM": 0.10, "FB": 0.05},
        "midfield": {"CM": 0.35, "DM": 0.25, "AM": 0.20, "WM": 0.10, "FB": 0.10},
        "defense": {"CB": 0.40, "FB": 0.25, "DM": 0.25, "GK": 0.10},
        "goalkeeper": {"GK": 1.00},
    },
    "attack_heavy": {
        "attack": {"ST": 0.50, "W": 0.30, "AM": 0.15, "CM": 0.05},
        "midfield": {"CM": 0.35, "DM": 0.25, "AM": 0.25, "WM": 0.10, "FB": 0.05},
        "defense": {"CB": 0.40, "FB": 0.25, "DM": 0.25, "GK": 0.10},
        "goalkeeper": {"GK": 1.00},
    },
    "midfield_control": {
        "attack": {"ST": 0.35, "W": 0.20, "AM": 0.25, "CM": 0.15, "FB": 0.05},
        "midfield": {"CM": 0.45, "DM": 0.30, "AM": 0.15, "WM": 0.05, "FB": 0.05},
        "defense": {"CB": 0.35, "FB": 0.20, "DM": 0.35, "GK": 0.10},
        "goalkeeper": {"GK": 1.00},
    },
    "defense_core": {
        "attack": {"ST": 0.35, "W": 0.25, "AM": 0.20, "CM": 0.10, "FB": 0.10},
        "midfield": {"CM": 0.30, "DM": 0.30, "AM": 0.15, "WM": 0.10, "FB": 0.15},
        "defense": {"CB": 0.45, "FB": 0.25, "DM": 0.20, "GK": 0.10},
        "goalkeeper": {"GK": 1.00},
    },
}


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def top_n_mean(values: list[float], n: int) -> float:
    if not values:
        return 0.0

    return mean(sorted(values, reverse=True)[:n])


def build_team_scores(players, projection: dict[str, dict[str, float]]) -> dict:
    attack_values = [
        weighted_role_score(player.role_ratings, projection["attack"])
        for player in players
    ]
    midfield_values = [
        weighted_role_score(player.role_ratings, projection["midfield"])
        for player in players
    ]
    defense_values = [
        weighted_role_score(player.role_ratings, projection["defense"])
        for player in players
    ]
    goalkeeper_values = [
        weighted_role_score(player.role_ratings, projection["goalkeeper"])
        for player in players
    ]

    overall_values = [
        player.ratings.overall
        for player in players
        if player.ratings.overall > 0
    ]

    evidence_values = [
        1.0
        for player in players
        if player.ratings.overall > 0
    ]

    return {
        "attack": top_n_mean(attack_values, 5),
        "midfield": top_n_mean(midfield_values, 5),
        "defense": top_n_mean(defense_values, 5),
        "goalkeeper": max(goalkeeper_values) if goalkeeper_values else 0.0,
        "squad_quality": mean(overall_values),
        "evidence_score": len(evidence_values) / len(players) if players else 0.0,
    }


def summarize_projection(rows: list[dict], projection_name: str) -> dict:
    df = pd.DataFrame(rows)

    result = {
        "projection": projection_name,
        "teams": len(df),
    }

    for feature in [
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "squad_quality",
        "evidence_score",
    ]:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()

        result[f"{feature}_mean"] = values.mean()
        result[f"{feature}_std"] = values.std()
        result[f"{feature}_min"] = values.min()
        result[f"{feature}_median"] = values.median()
        result[f"{feature}_max"] = values.max()

    return result


def main() -> None:
    roster_builder = RosterBuilder()
    teams = roster_builder.list_teams()

    summary_rows = []

    for projection_name, projection in PROJECTIONS.items():
        print(f"Evaluating projection: {projection_name}")

        team_rows = []

        for team in teams:
            squad = roster_builder.get_squad(team)

            if not squad.players:
                continue

            scores = build_team_scores(
                players=squad.players,
                projection=projection,
            )

            team_rows.append(
                {
                    "team": team,
                    **scores,
                }
            )

        summary_rows.append(
            summarize_projection(
                rows=team_rows,
                projection_name=projection_name,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Role Projection Comparison")
    print("--------------------------")
    print(summary.round(6).to_string(index=False))
    print()
    print(f"Wrote comparison -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()