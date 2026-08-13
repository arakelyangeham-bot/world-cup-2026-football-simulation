#validate_stage_resolver.py

from simulation.competition import (
    MatchResult,
    Stage,
    StageResolver,
    StageType,
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
    )

    stage.matches = [
        MatchResult("Argentina", "Japan", 2, 1),
        MatchResult("Morocco", "Canada", 1, 1),
        MatchResult("Argentina", "Morocco", 0, 0),
        MatchResult("Japan", "Canada", 3, 2),
        MatchResult("Argentina", "Canada", 4, 1),
        MatchResult("Japan", "Morocco", 1, 1),
    ]

    resolver = StageResolver()
    stage_result = resolver.resolve(stage)

    print(f"Stage: {stage_result.stage_name}")
    print(f"Resolved by: {stage_result.metadata['engine']}")
    print(f"Matches: {stage_result.metadata['match_count']}")
    print()

    for row in stage_result.standings.as_rows():
        print(
            f"{row['rank']}. {row['team']:<12} "
            f"{row['points']} pts "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )


if __name__ == "__main__":
    main()