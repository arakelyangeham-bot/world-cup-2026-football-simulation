#validate_wc_group_stage_framework_prototype.py

from simulation.competition import (
    MatchResult,
    Stage,
    StageResolver,
    StageType,
    TopNAdvanceRule,
)


def build_group_match_results(group_name: str, teams: list[str]) -> list[MatchResult]:
    """
    Build deterministic fake group-stage results.

    This does not simulate realistic football. It only validates whether a
    World Cup-style group can be expressed through the generic framework.
    """
    results: list[MatchResult] = []

    scorelines = [
        (2, 0),
        (1, 1),
        (2, 1),
        (0, 0),
        (3, 1),
        (1, 2),
    ]

    pairings = [
        (teams[0], teams[1]),
        (teams[2], teams[3]),
        (teams[0], teams[2]),
        (teams[1], teams[3]),
        (teams[0], teams[3]),
        (teams[1], teams[2]),
    ]

    for index, ((team1, team2), (goals1, goals2)) in enumerate(
        zip(pairings, scorelines),
        start=1,
    ):
        results.append(
            MatchResult(
                team1=team1,
                team2=team2,
                goals_team1=goals1,
                goals_team2=goals2,
                stage=group_name,
                match_id=f"{group_name}_match_{index}",
            )
        )

    return results


def main() -> None:
    group_name = "Group A"

    teams = [
        "Mexico",
        "South Africa",
        "South Korea",
        "UEFA Playoff Winner",
    ]

    group_stage = Stage(
        name=group_name,
        stage_type=StageType.GROUP,
        participants=teams,
        matches=build_group_match_results(group_name, teams),
        advancement_rule=TopNAdvanceRule(n=2),
        metadata={
            "competition": "FIFA World Cup 2026",
            "group": group_name,
            "study": "023",
        },
    )

    resolver = StageResolver()
    stage_result = resolver.resolve(group_stage)
    advancement_result = group_stage.advancement_rule.apply(stage_result)

    print(f"Stage: {stage_result.stage_name}")
    print(f"Engine: {stage_result.metadata['engine']}")
    print(f"Matches: {stage_result.metadata['match_count']}")
    print()

    print("Standings")
    print("---------")
    for row in stage_result.standings.as_rows():
        print(
            f"{row['rank']}. {row['team']:<22} "
            f"{row['points']} pts "
            f"W {row['wins']} "
            f"D {row['draws']} "
            f"L {row['losses']} "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )

    print()
    print("Top Two Qualifiers")
    print("------------------")
    for team in advancement_result.qualifiers:
        print(team)

    print()
    print("Eliminated")
    print("----------")
    for team in advancement_result.eliminated:
        print(team)


if __name__ == "__main__":
    main()