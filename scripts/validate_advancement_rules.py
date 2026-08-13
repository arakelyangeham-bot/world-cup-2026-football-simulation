#validate_advancement_rules.py

from simulation.competition import (
    MatchResult,
    Stage,
    StageType,
    StandingsEngine,
    TopNAdvanceRule,
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

    match_results = [
        MatchResult("Argentina", "Japan", 2, 1),
        MatchResult("Morocco", "Canada", 1, 1),
        MatchResult("Argentina", "Morocco", 0, 0),
        MatchResult("Japan", "Canada", 3, 2),
        MatchResult("Argentina", "Canada", 4, 1),
        MatchResult("Japan", "Morocco", 1, 1),
    ]

    engine = StandingsEngine()
    stage_result = engine.resolve(
        stage=stage,
        match_results=match_results,
    )

    advancement_rule = TopNAdvanceRule(n=2)
    advancement_result = advancement_rule.apply(stage_result)

    print(f"Stage: {stage_result.stage_name}")
    print(f"Rule: {advancement_result.metadata['rule']}")
    print()

    print("Standings")
    print("---------")
    for row in stage_result.standings.as_rows():
        print(
            f"{row['rank']}. {row['team']:<12} "
            f"{row['points']} pts "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )

    print()
    print("Qualifiers")
    print("----------")
    for team in advancement_result.qualifiers:
        print(team)

    print()
    print("Eliminated")
    print("----------")
    for team in advancement_result.eliminated:
        print(team)


if __name__ == "__main__":
    main()