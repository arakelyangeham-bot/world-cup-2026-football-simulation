#compare_aggregation_strategies.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.aggregation_strategies import (
    star_weighted,
    starter_plus_depth,
    top_n_mean,
    uniform_mean,
)
from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.player_representation_engine import (
    build_player_representation,
)
from research.player_intelligence.roster_builder import RosterBuilder


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "study_011_team_representation_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "aggregation_strategy_comparison.csv"


STRATEGIES = {
    "uniform_mean": uniform_mean,
    "top_11_mean": lambda players: top_n_mean(players, n=11),
    "top_5_mean": lambda players: top_n_mean(players, n=5),
    "starter_plus_depth": starter_plus_depth,
    "star_weighted": star_weighted,
}


def summarize(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()

    return {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "p75": values.quantile(0.75),
        "max": values.max(),
        "range": values.max() - values.min(),
    }


def main() -> None:
    evidence_repository = PlayerEvidenceRepository()

    player_repository = PlayerRepository(
        evidence_repository=evidence_repository,
    )

    roster_builder = RosterBuilder(repository=player_repository)

    teams = roster_builder.list_teams()

    rows = []

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"Evaluating aggregation strategy: {strategy_name}")

        team_values = []

        for team in teams:
            squad = roster_builder.get_squad(team)

            if not squad.players:
                continue

            player_representations = [
                build_player_representation(player)
                for player in squad.players
            ]

            team_score = strategy_fn(player_representations)

            team_values.append(
                {
                    "team": team,
                    "team_score": team_score,
                }
            )

        strategy_df = pd.DataFrame(team_values)
        summary = summarize(strategy_df["team_score"])

        rows.append(
            {
                "strategy": strategy_name,
                "teams": len(strategy_df),
                **summary,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Aggregation Strategy Comparison")
    print("-------------------------------")
    print(comparison.round(6).to_string(index=False))
    print()
    print(f"Wrote comparison -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()