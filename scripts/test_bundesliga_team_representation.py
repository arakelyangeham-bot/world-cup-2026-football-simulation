#test_bundesliga_team_representation

from __future__ import annotations

from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)


COMPETITION_ID = 35
SEASON_ID = 63516
TARGET_TEAM_NAME = "FC Bayern München"


def main() -> None:
    repository = CompetitionTeamRepository()

    teams = repository.roster_builder.list_teams(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
    )

    print("Bundesliga 2024/25 teams")
    print("========================")
    print(teams.to_string(index=False))
    print()

    matching = teams[
        teams["team"]
        .astype(str)
        .str.casefold()
        .eq(TARGET_TEAM_NAME.casefold())
    ]

    if matching.empty:
        available = teams["team"].astype(str).tolist()

        raise KeyError(
            f"Could not find {TARGET_TEAM_NAME!r}. "
            f"Available teams: {available}"
        )

    if len(matching) != 1:
        raise ValueError(
            f"Expected one match for {TARGET_TEAM_NAME!r}, "
            f"but found {len(matching)}."
        )

    team_id = int(
        matching.iloc[0]["team_id"]
    )

    print("Selected team")
    print("=============")
    print(f"Team: {TARGET_TEAM_NAME}")
    print(f"Team ID: {team_id}")
    print()

    context = repository.get_context(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        team_id=team_id,
    )

    print("Competition squad context")
    print("=========================")
    print(context)
    print()

    squad = repository.roster_builder.get_squad(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID,
        team_id=team_id,
        require_complete_join=True,
    )

    print("Squad construction")
    print("==================")
    print(f"Players joined: {len(squad.players)}")
    print()

    representation = (
        repository.get_full_squad_representation(
            competition_id=COMPETITION_ID,
            season_id=SEASON_ID,
            team_id=team_id,
        )
    )

    print("Full-squad representation")
    print("=========================")
    print(representation)
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()