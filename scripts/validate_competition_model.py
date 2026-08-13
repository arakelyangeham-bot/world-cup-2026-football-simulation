#validate_competition_model.py

from simulation.competition import (
    Competition,
    MatchResult,
    Stage,
    StageType,
)


def main() -> None:
    participants = [
        "Argentina",
        "Japan",
        "Morocco",
        "Canada",
    ]

    competition = Competition(
        name="Example Mini Competition",
        participants=participants,
        metadata={
            "study": "016",
            "purpose": "validate competition composition model",
        },
    )

    group_stage = Stage(
        name="Example Group",
        stage_type=StageType.GROUP,
        participants=participants,
    )

    group_stage.matches = [
        MatchResult("Argentina", "Japan", 2, 1),
        MatchResult("Morocco", "Canada", 1, 1),
        MatchResult("Argentina", "Morocco", 0, 0),
        MatchResult("Japan", "Canada", 3, 2),
        MatchResult("Argentina", "Canada", 4, 1),
        MatchResult("Japan", "Morocco", 1, 1),
    ]

    competition.add_stage(group_stage)

    print(f"Competition: {competition.name}")
    print(f"Participants: {len(competition.participants)}")
    print(f"Stages: {len(competition.stages)}")
    print()

    for stage in competition.stages:
        print(f"Stage: {stage.name}")
        print(f"Type: {stage.stage_type.value}")
        print(f"Participants: {len(stage.participants)}")
        print(f"Matches: {len(stage.matches)}")


if __name__ == "__main__":
    main()