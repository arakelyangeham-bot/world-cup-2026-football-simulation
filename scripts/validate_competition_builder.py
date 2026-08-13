#validate_competition_builder

from competition_catalog import CompetitionBuilder
from competition_catalog.competitions import PREMIER_LEAGUE


PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]


def main() -> None:
    builder = CompetitionBuilder()

    competition = builder.build(
        definition=PREMIER_LEAGUE,
        participants=PREMIER_LEAGUE_TEAMS,
    )

    print("Built competition")
    print("-----------------")
    print(competition.name)
    print(f"Participants: {len(competition.participants)}")
    print(f"Stages: {len(competition.stages)}")
    print()

    for stage in competition.stages:
        print(f"Stage: {stage.name}")
        print(f"Type: {stage.stage_type}")
        print(f"Participants: {len(stage.participants)}")
        print(f"Matches currently attached: {len(stage.matches)}")
        print(f"Metadata: {stage.metadata}")


if __name__ == "__main__":
    main()