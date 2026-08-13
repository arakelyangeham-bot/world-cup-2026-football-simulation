#build_player_intelligence_repository_by_strategy.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.aggregation_strategies import (
    star_weighted,
    starter_plus_depth,
    top_n_mean,
)
from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.player_representation_engine import (
    build_player_representation,
)
from research.player_intelligence.roster_builder import RosterBuilder
from shared.national_team_priors import load_fifa_points
from shared.team_name_normalizer import normalize_team_name

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
    / "repositories"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STRATEGIES = {
    "starter_plus_depth": starter_plus_depth,
    "top_11_mean": lambda players: top_n_mean(players, n=11),
    "top_5_mean": lambda players: top_n_mean(players, n=5),
    "star_weighted": star_weighted,
}


def build_repository_for_strategy(strategy_name: str, strategy_fn) -> pd.DataFrame:
    evidence_repository = PlayerEvidenceRepository()

    player_repository = PlayerRepository(
        evidence_repository=evidence_repository,
    )

    roster_builder = RosterBuilder(repository=player_repository)
    fifa_lookup = load_fifa_points()

    rows = []

    for team in roster_builder.list_teams():
        squad = roster_builder.get_squad(team)

        if not squad.players:
            continue

        player_representations = [
            build_player_representation(player)
            for player in squad.players
        ]

        team_score = strategy_fn(player_representations)
        canonical_team = normalize_team_name(team)
        fifa_points = fifa_lookup.get(canonical_team)
        
        if pd.isna(fifa_points):
            print(f"Skipping {canonical_team}: missing FIFA points.")
            continue

        rows.append(
            {
                "nation": canonical_team,
                "att_composite": team_score,
                "mid_composite": team_score,
                "def_composite": team_score,
                "gk_composite": team_score,
                "poisson_attack_adj": team_score,
                "poisson_defense_adj": team_score,
                "aggregation_strategy": strategy_name,
                "players": len(squad.players),
                "fifa_points": fifa_points,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"Building repository for strategy: {strategy_name}")

        repository = build_repository_for_strategy(
            strategy_name=strategy_name,
            strategy_fn=strategy_fn,
        )

        output_path = OUTPUT_DIR / f"{strategy_name}_team_repository.csv"
        repository.to_csv(output_path, index=False)

        print(f"  Teams: {len(repository)}")
        print(f"  Wrote -> {output_path}")

    print()
    print("Built sandbox Player Intelligence repositories.")


if __name__ == "__main__":
    main()