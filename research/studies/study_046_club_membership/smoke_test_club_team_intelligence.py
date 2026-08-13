#smoke_test_club_team_intelligence

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.player_intelligence.competition_roster_builder import (
    CompetitionRosterBuilder,
)
from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
)
from research.player_intelligence.team_representation_builder import (
    build_team_representation_from_starting_xi,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FORMATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)


def select_first_club_context(
    roster_builder: CompetitionRosterBuilder,
) -> tuple[int, int, int]:
    seasons = (
        roster_builder
        .list_competition_seasons()
    )

    club_seasons = seasons[
        seasons["competition_type"]
        .astype(str)
        .str.contains(
            "club",
            case=False,
            na=False,
        )
    ]

    if club_seasons.empty:
        raise RuntimeError(
            "No club competition-seasons were found."
        )

    season = club_seasons.iloc[0]

    competition_id = int(
        season["competition_id"]
    )
    season_id = int(
        season["season_id"]
    )

    teams = roster_builder.list_teams(
        competition_id=competition_id,
        season_id=season_id,
    )

    if teams.empty:
        raise RuntimeError(
            "Selected club competition-season "
            "contains no teams."
        )

    team_id = int(
        teams.iloc[0]["team_id"]
    )

    return (
        competition_id,
        season_id,
        team_id,
    )


def load_formation() -> pd.DataFrame:
    if not FORMATION_FILE.exists():
        raise FileNotFoundError(
            "Formation manifest does not exist: "
            f"{FORMATION_FILE}"
        )

    formation_df = pd.read_csv(
        FORMATION_FILE
    )

    if "formation" in formation_df.columns:
        formation_df = formation_df[
            formation_df["formation"]
            .astype(str)
            .eq("4-3-3")
        ].copy()

    if formation_df.empty:
        raise ValueError(
            "No 4-3-3 formation rows were found."
        )

    return formation_df.reset_index(
        drop=True
    )


def main() -> None:
    roster_builder = (
        CompetitionRosterBuilder()
    )

    (
        competition_id,
        season_id,
        team_id,
    ) = select_first_club_context(
        roster_builder
    )

    context = roster_builder.get_context(
        competition_id=competition_id,
        season_id=season_id,
        team_id=team_id,
    )

    squad = roster_builder.get_squad(
        competition_id=competition_id,
        season_id=season_id,
        team_id=team_id,
    )

    formation_df = load_formation()

    lineup_builder = StartingXIBuilder(
        formation="4-3-3"
    )

    starting_xi = (
        lineup_builder.build_for_squad(
            squad=squad,
            formation_df=formation_df,
        )
    )

    representation = (
        build_team_representation_from_starting_xi(
            starting_xi
        )
    )

    print("Club Team Intelligence Smoke Test")
    print("================================")
    print()
    print(
        f"Competition: {context.competition}"
    )
    print(
        f"Season: {context.season_year}"
    )
    print(
        f"Team: {context.team}"
    )
    print(
        f"Squad players: {len(squad.players)}"
    )
    print(
        f"Formation: {starting_xi.formation}"
    )
    print(
        "Starting XI players: "
        f"{len(starting_xi.players)}"
    )
    print()

    lineup_rows = []

    for slot, player in zip(
        formation_df["slot"],
        starting_xi.players,
    ):
        lineup_rows.append(
            {
                "slot": slot,
                "player_id": (
                    player.identity.player_id
                ),
                "player": player.identity.name,
                "primary_position": (
                    player.identity.primary_position
                ),
            }
        )

    print(
        pd.DataFrame(lineup_rows)
        .to_string(index=False)
    )

    print()
    print("Team Representation")
    print("-------------------")
    print(
        "Representation type: "
        f"{representation.representation_type}"
    )
    print(
        "Aggregation profile: "
        f"{representation.aggregation_profile}"
    )
    print(
        f"Player count: "
        f"{representation.player_count}"
    )
    print(
        f"Available players: "
        f"{representation.available_player_count}"
    )
    print()
    print(
        f"Attack: {representation.attack:.6f}"
    )
    print(
        f"Midfield: "
        f"{representation.midfield:.6f}"
    )
    print(
        f"Defense: "
        f"{representation.defense:.6f}"
    )
    print(
        f"Goalkeeper: "
        f"{representation.goalkeeper:.6f}"
    )
    print(
        f"Squad quality: "
        f"{representation.squad_quality:.6f}"
    )
    print(
        f"Evidence score: "
        f"{representation.evidence_score:.6f}"
    )

    if len(starting_xi.players) != len(
        formation_df
    ):
        raise AssertionError(
            "Starting XI player count does not "
            "match formation slots."
        )

    if representation.player_count != len(
        starting_xi.players
    ):
        raise AssertionError(
            "Representation player count does "
            "not match Starting XI."
        )

    if (
        representation.representation_type
        != "expected_starting_xi"
    ):
        raise AssertionError(
            "Unexpected representation type: "
            f"{representation.representation_type}"
        )

    print()
    print("Squad → StartingXI: PASS")
    print(
        "StartingXI → TeamRepresentation: PASS"
    )
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()