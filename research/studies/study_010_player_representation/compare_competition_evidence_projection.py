#compare_competition_evidence_projection.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.competition_evidence_projection import (
    combined_competition_weight,
    competition_count_weight,
    identity_competition_weight,
    season_count_weight,
    weighted_evidence_weight,
)
from research.player_intelligence.roster_builder import RosterBuilder
from research.player_intelligence.role_projection import weighted_role_score


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "study_010_player_representation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "competition_evidence_projection_comparison.csv"


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


def competition_weight_for_strategy(player, strategy: str) -> float:
    evidence = player.evidence

    if strategy == "identity":
        return identity_competition_weight()

    if strategy == "competition_count":
        return competition_count_weight(evidence.competition_count)

    if strategy == "season_count":
        return season_count_weight(evidence.season_count)

    if strategy == "weighted_evidence":
        return weighted_evidence_weight(evidence.total_weighted_evidence)

    if strategy == "combined_competition":
        return combined_competition_weight(
            competition_count=evidence.competition_count,
            season_count=evidence.season_count,
            total_weighted_evidence=evidence.total_weighted_evidence,
        )

    raise ValueError(f"Unknown competition evidence strategy: {strategy}")


def build_team_scores(players, strategy: str) -> dict:
    attack_values = []
    midfield_values = []
    defense_values = []
    goalkeeper_values = []
    evidence_weights = []

    for player in players:
        evidence_weight = competition_weight_for_strategy(player, strategy)
        evidence_weights.append(evidence_weight)

        attack_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["attack"])
            * evidence_weight
        )
        midfield_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["midfield"])
            * evidence_weight
        )
        defense_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["defense"])
            * evidence_weight
        )
        goalkeeper_values.append(
            weighted_role_score(player.role_ratings, ROLE_WEIGHTS["goalkeeper"])
            * evidence_weight
        )

    return {
        "attack": top_n_mean(attack_values, 5),
        "midfield": top_n_mean(midfield_values, 5),
        "defense": top_n_mean(defense_values, 5),
        "goalkeeper": max(goalkeeper_values) if goalkeeper_values else 0.0,
        "mean_competition_evidence_weight": mean(evidence_weights),
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
        "mean_competition_evidence_weight",
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
        "competition_count",
        "season_count",
        "weighted_evidence",
        "combined_competition",
    ]

    summary_rows = []

    for strategy in strategies:
        print(f"Evaluating competition evidence strategy: {strategy}")

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
    print("Competition Evidence Projection Comparison")
    print("------------------------------------------")
    print(summary.round(6).to_string(index=False))
    print()
    print(f"Wrote comparison -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()