#validate_knockout_engine.py

from simulation.competition import (
    KnockoutEngine,
    MatchResult,
    Stage,
    StageType,
    Tie,
)


def main() -> None:
    stage = Stage(
        name="Example Semifinals",
        stage_type=StageType.KNOCKOUT,
        participants=[
            "Argentina",
            "Japan",
            "Brazil",
            "Morocco",
        ],
    )

    tie_1 = Tie(
        team1="Argentina",
        team2="Japan",
        match_results=[
            MatchResult("Argentina", "Japan", 2, 0),
        ],
    )

    tie_2 = Tie(
        team1="Brazil",
        team2="Morocco",
        match_results=[
            MatchResult("Brazil", "Morocco", 1, 2),
        ],
    )

    stage.matches = [tie_1, tie_2]

    engine = KnockoutEngine()
    result = engine.resolve(
        stage=stage,
        match_results=stage.matches,
    )

    print(f"Stage: {result.stage_name}")
    print(f"Engine: {result.metadata['engine']}")
    print(f"Ties: {result.metadata['tie_count']}")
    print()

    print("Qualifiers")
    print("----------")
    for team in result.qualifiers:
        print(team)

    print()
    print("Eliminated")
    print("----------")
    for team in result.eliminated:
        print(team)


if __name__ == "__main__":
    main()