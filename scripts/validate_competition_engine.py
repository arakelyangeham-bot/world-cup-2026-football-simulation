#validate_competition_engine.py

from simulation.competition import (
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
    TopNAdvanceRule,
)


def main() -> None:
    participants = [
        "Argentina",
        "Japan",
        "Morocco",
        "Canada",
    ]

    group_stage = Stage(
        name="Example Group",
        stage_type=StageType.GROUP,
        participants=participants,
        advancement_rule=TopNAdvanceRule(n=2),
    )

    group_stage.matches = [
        MatchResult("Argentina", "Japan", 2, 1),
        MatchResult("Morocco", "Canada", 1, 1),
        MatchResult("Argentina", "Morocco", 0, 0),
        MatchResult("Japan", "Canada", 3, 2),
        MatchResult("Argentina", "Canada", 4, 1),
        MatchResult("Japan", "Morocco", 1, 1),
    ]

    competition = Competition(
        name="Example Mini Competition",
        participants=participants,
        stages=[group_stage],
    )

    engine = CompetitionEngine()
    result = engine.resolve(competition)

    print(f"Competition: {result.competition_name}")
    print(f"Engine: {result.metadata['engine']}")
    print(f"Stages resolved: {len(result.stage_results)}")
    print()

    stage_result = result.stage_results[0]
    advancement_result = result.advancement_results[0]

    print(f"Stage: {stage_result.stage_name}")
    print(f"Stage engine: {stage_result.metadata['engine']}")
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

    print()
    print(f"Simple champion: {result.champion}")
    print(f"Simple runner-up: {result.runner_up}")


if __name__ == "__main__":
    main()