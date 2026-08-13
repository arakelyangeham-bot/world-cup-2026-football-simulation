#build_dimension_specific_team_repository.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.dimension_aggregation import (
    aggregate_team_dimensions,
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

LEGACY_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_team_strength.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
    / "repositories"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "dimension_specific_team_repository.csv"


def main() -> None:
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

        dimensions = aggregate_team_dimensions(player_representations)
        canonical_team = normalize_team_name(team)
        fifa_points = fifa_lookup.get(canonical_team)

        if pd.isna(fifa_points):
            print(f"Skipping {canonical_team}: missing FIFA points.")
            continue

        rows.append(
            {
                "nation": canonical_team,
                "att_composite": dimensions["attack"],
                "mid_composite": dimensions["midfield"],
                "def_composite": dimensions["defense"],
                "gk_composite": dimensions["goalkeeper"],
                "poisson_attack_adj": dimensions["attack"],
                "poisson_defense_adj": dimensions["defense"],
                "aggregation_strategy": "dimension_specific",
                "players": len(squad.players),
                "fifa_points": fifa_points,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("Dimension-Specific Team Repository Build")
    print("----------------------------------------")
    print(f"Teams: {len(out)}")
    print(out.head(20).round(4).to_string(index=False))
    print()
    print(f"Wrote -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()