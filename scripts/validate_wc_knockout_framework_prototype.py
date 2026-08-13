#validate_wc_knockout_framework_prototype.py

from simulation.competition import (
    BracketBuilder,
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
)


QUALIFIED_TEAMS = [
    "Mexico",
    "South Korea",
    "Canada",
    "Switzerland",
    "Brazil",
    "Scotland",
    "United States",
    "Australia",
    "Germany",
    "Ivory Coast",
    "Netherlands",
    "European Playoff Winner",
    "Belgium",
    "Iran",
    "Spain",
    "Saudi Arabia",
    "France",
    "Norway",
    "Argentina",
    "Austria",
    "Portugal",
    "Colombia",
    "England",
    "Panama",
    "Morocco",
    "Japan",
    "Uruguay",
    "Croatia",
    "Senegal",
    "Ecuador",
    "Ghana",
    "Tunisia",
]


def attach_placeholder_results(ties, round_name: str) -> None:
    """
    Attach deterministic non-draw placeholder results to knockout ties.

    This is not intended to be realistic. It only validates that the knockout
    framework can resolve a full World Cup-style knockout phase.
    """

    for index, tie in enumerate(ties, start=1):
        if index % 3 == 0:
            goals_team1, goals_team2 = 1, 2
        elif index % 3 == 1:
            goals_team1, goals_team2 = 2, 0
        else:
            goals_team1, goals_team2 = 3, 1

        tie.add_match_result(
            MatchResult(
                team1=tie.team1,
                team2=tie.team2,
                goals_team1=goals_team1,
                goals_team2=goals_team2,
                stage=round_name,
                match_id=f"{round_name}_tie_{index}",
                metadata={
                    "study": "025",
                    "placeholder_result": True,
                },
            )
        )


def build_knockout_stage(
    round_name: str,
    teams: list[str],
    stage_type: StageType = StageType.KNOCKOUT,
) -> Stage:
    builder = BracketBuilder()
    bracket = builder.build_high_low_bracket(
        teams=teams,
        name=f"{round_name} Bracket",
    )

    attach_placeholder_results(
        ties=bracket.ties,
        round_name=round_name,
    )

    return Stage(
        name=round_name,
        stage_type=stage_type,
        participants=teams,
        matches=bracket.as_stage_matches(),
        metadata={
            "study": "025",
            "bracket": bracket.name,
            "pairing_type": bracket.metadata["pairing_type"],
        },
    )


def resolve_stage(engine: CompetitionEngine, stage: Stage):
    temp_competition = Competition(
        name=f"{stage.name} Temporary Competition",
        participants=stage.participants,
        stages=[stage],
    )

    result = engine.resolve(temp_competition)
    return result.stage_results[0]


def print_stage_summary(stage_result) -> None:
    print(stage_result.stage_name)
    print("-" * len(stage_result.stage_name))
    print(f"Engine: {stage_result.metadata['engine']}")
    print(f"Ties: {stage_result.metadata['tie_count']}")
    print(f"Qualifiers: {', '.join(stage_result.qualifiers)}")
    print(f"Eliminated: {', '.join(stage_result.eliminated)}")

    if stage_result.winner:
        print(f"Winner: {stage_result.winner}")
        print(f"Runner-up: {stage_result.runner_up}")

    print()


def main() -> None:
    engine = CompetitionEngine()

    stage_results = []

    round_of_32 = build_knockout_stage(
        round_name="Round of 32",
        teams=QUALIFIED_TEAMS,
    )
    r32_result = resolve_stage(engine, round_of_32)
    stage_results.append(r32_result)

    round_of_16 = build_knockout_stage(
        round_name="Round of 16",
        teams=r32_result.qualifiers,
    )
    r16_result = resolve_stage(engine, round_of_16)
    stage_results.append(r16_result)

    quarterfinals = build_knockout_stage(
        round_name="Quarterfinals",
        teams=r16_result.qualifiers,
    )
    qf_result = resolve_stage(engine, quarterfinals)
    stage_results.append(qf_result)

    semifinals = build_knockout_stage(
        round_name="Semifinals",
        teams=qf_result.qualifiers,
    )
    sf_result = resolve_stage(engine, semifinals)
    stage_results.append(sf_result)

    third_place_playoff = build_knockout_stage(
        round_name="Third-place Playoff",
        teams=sf_result.eliminated,
        stage_type=StageType.PLAYOFF,
    )
    third_place_result = resolve_stage(engine, third_place_playoff)
    stage_results.append(third_place_result)

    final = build_knockout_stage(
        round_name="Final",
        teams=sf_result.qualifiers,
        stage_type=StageType.FINAL,
    )
    final_result = resolve_stage(engine, final)
    stage_results.append(final_result)

    competition = Competition(
        name="Framework-backed World Cup Knockout Prototype",
        participants=QUALIFIED_TEAMS,
        stages=[
            round_of_32,
            round_of_16,
            quarterfinals,
            semifinals,
            third_place_playoff,
            final,
        ],
        metadata={
            "study": "025",
            "prototype": "world_cup_knockout_framework",
        },
    )

    competition_result = engine.resolve(competition)

    print("Framework-backed World Cup Knockout Prototype")
    print("=============================================")
    print()

    for stage_result in stage_results:
        print_stage_summary(stage_result)

    print("Summary")
    print("-------")
    print(f"Initial teams: {len(QUALIFIED_TEAMS)}")
    print(f"Stages resolved: {len(stage_results)}")
    print(f"Champion: {final_result.winner}")
    print(f"Runner-up: {final_result.runner_up}")
    print(f"Third place: {third_place_result.winner}")
    print(f"Fourth place: {third_place_result.runner_up}")


if __name__ == "__main__":
    main()