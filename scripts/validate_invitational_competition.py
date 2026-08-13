#validate_invitational_competition.py

from simulation.competition import (
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
    Tie,
)


def main() -> None:
    participants = [
        "Argentina",
        "Japan",
        "Morocco",
        "Canada",
    ]

    semifinals = Stage(
        name="Semifinals",
        stage_type=StageType.KNOCKOUT,
        participants=participants,
    )

    semifinals.matches = [
        Tie(
            team1="Argentina",
            team2="Canada",
            match_results=[
                MatchResult("Argentina", "Canada", 3, 1),
            ],
        ),
        Tie(
            team1="Japan",
            team2="Morocco",
            match_results=[
                MatchResult("Japan", "Morocco", 1, 2),
            ],
        ),
    ]

    final = Stage(
        name="Final",
        stage_type=StageType.FINAL,
        participants=[
            "Argentina",
            "Morocco",
        ],
    )

    final.matches = [
        Tie(
            team1="Argentina",
            team2="Morocco",
            match_results=[
                MatchResult("Argentina", "Morocco", 2, 0),
            ],
        )
    ]

    competition = Competition(
        name="Example Invitational Cup",
        participants=participants,
        stages=[
            semifinals,
            final,
        ],
    )

    engine = CompetitionEngine()
    result = engine.resolve(competition)

    print(f"Competition: {result.competition_name}")
    print(f"Stages resolved: {len(result.stage_results)}")
    print()

    for stage_result in result.stage_results:
        print(f"Stage: {stage_result.stage_name}")
        print(f"Engine: {stage_result.metadata['engine']}")
        print(f"Qualifiers: {', '.join(stage_result.qualifiers)}")
        print(f"Eliminated: {', '.join(stage_result.eliminated)}")

        if stage_result.winner:
            print(f"Winner: {stage_result.winner}")
            print(f"Runner-up: {stage_result.runner_up}")

        print()

    print(f"Champion: {result.champion}")
    print(f"Runner-up: {result.runner_up}")


if __name__ == "__main__":
    main()