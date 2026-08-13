from competition_catalog import CompetitionCatalog
from competition_catalog.competitions import (
    PREMIER_LEAGUE,
    WORLD_CUP_2026,
)

def main() -> None:
    catalog = CompetitionCatalog()

    catalog.register(PREMIER_LEAGUE)
    catalog.register(WORLD_CUP_2026)

    print("Available competitions")
    print("----------------------")
    for name in catalog.names():
        print(name)

    print()

    competition = catalog.get("Premier League")
    print("Loaded competition")
    print("------------------")
    print(competition.name)
    print(competition.competition_type)
    print(competition.stage_names())

    print()

    world_cup = catalog.get("FIFA World Cup 2026")

    print("Loaded World Cup")
    print("----------------")
    print(world_cup.name)
    print(world_cup.competition_type)
    print(world_cup.participant_count)

    for stage in world_cup.stages:
        print(
            f"{stage.name}: "
            f"{stage.stage_type}, "
            f"{stage.participant_count} participants"
        )


if __name__ == "__main__":
    main()