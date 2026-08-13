#validate_league_competition_prototype.py

from simulation.competition import (
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
)


def build_single_round_robin_results(teams: list[str]) -> list[MatchResult]:
    """
    Build a deterministic single round-robin result set.

    This is not intended to be realistic. It only provides completed
    match results for validating the league competition framework.
    """
    results: list[MatchResult] = []

    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i >= j:
                continue

            goals_team1 = (i + j) % 4
            goals_team2 = (j - i) % 3

            results.append(
                MatchResult(
                    team1=team1,
                    team2=team2,
                    goals_team1=goals_team1,
                    goals_team2=goals_team2,
                    stage="League Stage",
                    match_id=f"{team1}_vs_{team2}",
                )
            )

    return results


def main() -> None:
    teams = [
        "Arsenal",
        "Liverpool",
        "Manchester City",
        "Chelsea",
        "Tottenham",
        "Newcastle United",
        "Aston Villa",
        "Brighton",
    ]

    league_stage = Stage(
        name="League Stage",
        stage_type=StageType.LEAGUE,
        participants=teams,
        matches=build_single_round_robin_results(teams),
        metadata={
            "format": "single_round_robin",
            "team_count": len(teams),
        },
    )

    competition = Competition(
        name="Example 8-Team League",
        participants=teams,
        stages=[
            league_stage,
        ],
        metadata={
            "study": "022",
            "format": "league_prototype",
        },
    )

    engine = CompetitionEngine()
    result = engine.resolve(competition)

    stage_result = result.stage_results[0]

    print(f"Competition: {result.competition_name}")
    print(f"Stages resolved: {len(result.stage_results)}")
    print(f"Matches: {stage_result.metadata['match_count']}")
    print()

    print("Final Table")
    print("-----------")
    for row in stage_result.standings.as_rows():
        print(
            f"{row['rank']}. {row['team']:<18} "
            f"{row['points']:>2} pts "
            f"W {row['wins']} "
            f"D {row['draws']} "
            f"L {row['losses']} "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )

    print()
    print(f"Champion: {result.champion}")
    print(f"Runner-up: {result.runner_up}")


if __name__ == "__main__":
    main()