#compare_recency_projection.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.recency_projection import (
    blended_recency_weight,
    exponential_recency_weight,
    identity_recency_weight,
    linear_recency_weight,
)
from research.player_intelligence.roster_builder import RosterBuilder
from research.player_intelligence.role_projection import weighted_role_score


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "study_010_player_representation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "recency_projection_comparison.csv"


ROLE_WEIGHTS = {
    "attack": {"ST": 0.40, "W": 0.25, "AM": 0.20, "CM": 0.10, "FB": 0.05},
    "midfield": {"CM": 0.35, "DM": 0.25, "AM": 0.20, "WM": 0.10, "FB": 0.10},
    "defense": {"CB": 0.40, "FB": 0.25, "DM": 0.25, "GK": 0.10},
    "goalkeeper": {"GK": 1.00},
}


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def top_n_mean(values: list[float], n: int) -> float:
    if not values:
        return 0.0
    return mean(sorted(values, reverse=True)[:n])


def recency_weight_for_strategy(player, strategy: str) -> float:
    recency_weight = player.evidence.recency_weight

    if strategy == "identity":
        return identity_recency_weight()

    if strategy == "linear":
        return linear_recency_weight(recency_weight)

    if strategy == "exponential":
        return exponential_recency_weight(recency_weight)

    if strategy == "blended":
        return blended_recency_weight(recency_weight)

    raise ValueError(f"Unknown recency strategy: {strategy}")


def build_team_scores(players, strategy: str) -> dict:
    attack_values = []
    midfield_values = []
    defense_values = []
    goalkeeper_values = []
    recency_weights = []

    for player in players:
        recency_weight = recency_weight_for_strategy(player, strategy)
        recency_weights.append(recency_weight)

        attack_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["attack"])
            * recency_weight
        )
        midfield_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["midfield"])
            * recency_weight
        )
        defense_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["defense"])
            * recency_weight
        )
        goalkeeper_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["goalkeeper"])
            * recency_weight
        )

    return {
        "attack": top_n_mean(attack_values, 5),
        "midfield": top_n_mean(midfield_values, 5),
        "defense": top_n_mean(defense_values, 5),
        "goalkeeper": max(goalkeeper_values) if goalkeeper_values else 0.0,
        "mean_recency_weight": mean(recency_weights),
    }


def summarize_strategy(rows: list[dict], strategy: str) -> dict:
    df = pd.DataFrame(rows)

    result = {
        "strategy": strategy,
        "teams": len(df),
    }

    for feature in [
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "mean_recency_weight",
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

    strategies = [
        "identity",
        "linear",
        "exponential",
        "blended",
    ]

    summary_rows = []

    for strategy in strategies:
        print(f"Evaluating recency strategy: {strategy}")

        team_rows = []

        for team in teams:
            squad = roster_builder.get_squad(team)

            if not squad.players:
                continue

            scores = build_team_scores(
                players=squad.players,
                strategy=strategy,
            )

            team_rows.append(
                {
                    "team": team,
                    **scores,
                }
            )

        summary_rows.append(
            summarize_strategy(
                rows=team_rows,
                strategy=strategy,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Recency Projection Comparison")
    print("-----------------------------")
    print(summary.round(6).to_string(index=False))
    print()
    print(f"Wrote comparison -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()