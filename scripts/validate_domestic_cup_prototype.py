#validate_domestic_cup_prototype.py

from simulation.competition import (
    BracketBuilder,
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
)


def attach_results_to_ties(ties, scorelines):
    for tie, (goals_team1, goals_team2) in zip(ties, scorelines):
        tie.add_match_result(
            MatchResult(
                team1=tie.team1,
                team2=tie.team2,
                goals_team1=goals_team1,
                goals_team2=goals_team2,
            )
        )


def build_knockout_stage(name, teams, scorelines):
    builder = BracketBuilder()
    bracket = builder.build_high_low_bracket(
        teams=teams,
        name=f"{name} Bracket",
    )

    attach_results_to_ties(bracket.ties, scorelines)

    return Stage(
        name=name,
        stage_type=StageType.KNOCKOUT,
        participants=teams,
        matches=bracket.as_stage_matches(),
        metadata={
            "bracket": bracket.name,
            "pairing_type": bracket.metadata["pairing_type"],
        },
    )


def main() -> None:
    quarterfinal_teams = [
        "Arsenal",
        "Liverpool",
        "Manchester City",
        "Chelsea",
        "Tottenham",
        "Newcastle United",
        "Aston Villa",
        "Brighton",
    ]

    quarterfinals = build_knockout_stage(
        name="Quarterfinals",
        teams=quarterfinal_teams,
        scorelines=[
            (2, 0),  # Arsenal vs Brighton
            (1, 2),  # Liverpool vs Aston Villa
            (3, 1),  # Manchester City vs Newcastle United
            (0, 1),  # Chelsea vs Tottenham
        ],
    )

    # Winners from quarterfinals:
    # Arsenal, Aston Villa, Manchester City, Tottenham
    semifinals = build_knockout_stage(
        name="Semifinals",
        teams=[
            "Arsenal",
            "Aston Villa",
            "Manchester City",
            "Tottenham",
        ],
        scorelines=[
            (2, 1),  # Arsenal vs Tottenham
            (3, 2),  # Aston Villa vs Manchester City
        ],
    )

    # Winners from semifinals:
    # Arsenal, Aston Villa
    final = build_knockout_stage(
        name="Final",
        teams=[
            "Arsenal",
            "Aston Villa",
        ],
        scorelines=[
            (2, 0),  # Arsenal vs Aston Villa
        ],
    )

    competition = Competition(
        name="Example Domestic Cup",
        participants=quarterfinal_teams,
        stages=[
            quarterfinals,
            semifinals,
            final,
        ],
        metadata={
            "study": "020",
            "format": "8-team knockout cup",
        },
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