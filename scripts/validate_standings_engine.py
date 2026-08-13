#validate_standings_engine.py

from simulation.competition import (
    MatchResult,
    Stage,
    StageType,
    StandingsEngine,
)


def main() -> None:
    stage = Stage(
        name="Example Group",
        stage_type=StageType.GROUP,
        participants=[
            "Argentina",
            "Japan",
            "Morocco",
            "Canada",
        ],
        metadata={
            "study": "014",
            "purpose": "validate generic standings engine",
        },
    )

    match_results = [
        MatchResult("Argentina", "Japan", 2, 1),
        MatchResult("Morocco", "Canada", 1, 1),
        MatchResult("Argentina", "Morocco", 0, 0),
        MatchResult("Japan", "Canada", 3, 2),
        MatchResult("Argentina", "Canada", 4, 1),
        MatchResult("Japan", "Morocco", 1, 1),
    ]

    engine = StandingsEngine()
    result = engine.resolve(
        stage=stage,
        match_results=match_results,
    )

    print(f"Stage: {result.stage_name}")
    print(f"Type: {result.stage_type}")
    print(f"Matches: {result.metadata['match_count']}")
    print()

    for row in result.standings.as_rows():
        print(
            f"{row['rank']}. {row['team']:<12} "
            f"{row['points']} pts "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )


if __name__ == "__main__":
    main()