#validate_complete_framework_backed_world_cup.py

from simulation.competition import (
    BracketBuilder,
    MatchResult,
    Stage,
    StageResolver,
    StageType,
    TopNAdvanceRule,
)


WORLD_CUP_GROUPS = {
    "Group A": [
        "Mexico",
        "South Africa",
        "South Korea",
        "UEFA Playoff Winner",
    ],
    "Group B": [
        "Canada",
        "Qatar",
        "Switzerland",
        "CAF Playoff Winner",
    ],
    "Group C": [
        "Brazil",
        "Morocco",
        "Scotland",
        "Haiti",
    ],
    "Group D": [
        "United States",
        "Paraguay",
        "Australia",
        "UEFA Playoff Winner 2",
    ],
    "Group E": [
        "Germany",
        "Curacao",
        "Ivory Coast",
        "Ecuador",
    ],
    "Group F": [
        "Netherlands",
        "Japan",
        "European Playoff Winner",
        "Tunisia",
    ],
    "Group G": [
        "Belgium",
        "Egypt",
        "Iran",
        "New Zealand",
    ],
    "Group H": [
        "Spain",
        "Cape Verde",
        "Saudi Arabia",
        "Uruguay",
    ],
    "Group I": [
        "France",
        "Senegal",
        "Norway",
        "Concacaf Playoff Winner",
    ],
    "Group J": [
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan",
    ],
    "Group K": [
        "Portugal",
        "Uzbekistan",
        "Colombia",
        "Ghana",
    ],
    "Group L": [
        "England",
        "Croatia",
        "Panama",
        "Saudi Arabia 2",
    ],
}


PLACEHOLDER_EXTRA_QUALIFIERS = [
    "Morocco",
    "Japan",
    "Uruguay",
    "Croatia",
    "Senegal",
    "Ecuador",
    "Ghana",
    "Tunisia",
]


def build_group_match_results(group_name: str, teams: list[str]) -> list[MatchResult]:
    pairings = [
        (teams[0], teams[1]),
        (teams[2], teams[3]),
        (teams[0], teams[2]),
        (teams[1], teams[3]),
        (teams[0], teams[3]),
        (teams[1], teams[2]),
    ]

    scorelines = [
        (2, 0),
        (1, 1),
        (2, 1),
        (0, 0),
        (3, 1),
        (1, 2),
    ]

    results: list[MatchResult] = []

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
                metadata={
                    "study": "026",
                    "placeholder_result": True,
                },
            )
        )

    return results


def build_group_stage(group_name: str, teams: list[str]) -> Stage:
    return Stage(
        name=group_name,
        stage_type=StageType.GROUP,
        participants=teams,
        matches=build_group_match_results(group_name, teams),
        advancement_rule=TopNAdvanceRule(n=2),
        metadata={
            "competition": "FIFA World Cup 2026",
            "group": group_name,
            "study": "026",
        },
    )


def attach_knockout_placeholder_results(ties, round_name: str) -> None:
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
                    "study": "026",
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

    attach_knockout_placeholder_results(
        ties=bracket.ties,
        round_name=round_name,
    )

    return Stage(
        name=round_name,
        stage_type=stage_type,
        participants=teams,
        matches=bracket.as_stage_matches(),
        metadata={
            "competition": "FIFA World Cup 2026",
            "study": "026",
            "bracket": bracket.name,
            "pairing_type": bracket.metadata["pairing_type"],
        },
    )


def resolve_stage(resolver: StageResolver, stage: Stage):
    stage_result = resolver.resolve(stage)

    advancement_result = None
    if stage.advancement_rule is not None:
        advancement_result = stage.advancement_rule.apply(stage_result)

    return stage_result, advancement_result


def print_group_summary(group_name, stage_result, advancement_result) -> None:
    print(group_name)
    print("-" * len(group_name))

    for row in stage_result.standings.as_rows():
        print(
            f"{row['rank']}. {row['team']:<28} "
            f"{row['points']} pts "
            f"W {row['wins']} "
            f"D {row['draws']} "
            f"L {row['losses']} "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )

    print()
    print("Automatic qualifiers:")
    for team in advancement_result.qualifiers:
        print(f"- {team}")

    print()


def print_knockout_summary(stage_result) -> None:
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
    resolver = StageResolver()

    automatic_qualifiers: list[str] = []
    group_stage_results = []

    print("Complete Framework-backed World Cup Prototype")
    print("============================================")
    print()

    print("Group Stage")
    print("===========")
    print()

    for group_name, teams in WORLD_CUP_GROUPS.items():
        group_stage = build_group_stage(group_name, teams)
        stage_result, advancement_result = resolve_stage(resolver, group_stage)

        group_stage_results.append(stage_result)
        automatic_qualifiers.extend(advancement_result.qualifiers)

        print_group_summary(
            group_name=group_name,
            stage_result=stage_result,
            advancement_result=advancement_result,
        )

    knockout_teams = automatic_qualifiers + PLACEHOLDER_EXTRA_QUALIFIERS

    print("Group Stage Summary")
    print("-------------------")
    print(f"Groups resolved: {len(WORLD_CUP_GROUPS)}")
    print(f"Group-stage matches: {len(WORLD_CUP_GROUPS) * 6}")
    print(f"Automatic qualifiers: {len(automatic_qualifiers)}")
    print(f"Placeholder extra qualifiers: {len(PLACEHOLDER_EXTRA_QUALIFIERS)}")
    print(f"Knockout teams: {len(knockout_teams)}")
    print()

    print("Knockout Stage")
    print("==============")
    print()

    round_of_32 = build_knockout_stage(
        round_name="Round of 32",
        teams=knockout_teams,
    )
    r32_result, _ = resolve_stage(resolver, round_of_32)
    print_knockout_summary(r32_result)

    round_of_16 = build_knockout_stage(
        round_name="Round of 16",
        teams=r32_result.qualifiers,
    )
    r16_result, _ = resolve_stage(resolver, round_of_16)
    print_knockout_summary(r16_result)

    quarterfinals = build_knockout_stage(
        round_name="Quarterfinals",
        teams=r16_result.qualifiers,
    )
    qf_result, _ = resolve_stage(resolver, quarterfinals)
    print_knockout_summary(qf_result)

    semifinals = build_knockout_stage(
        round_name="Semifinals",
        teams=qf_result.qualifiers,
    )
    sf_result, _ = resolve_stage(resolver, semifinals)
    print_knockout_summary(sf_result)

    third_place_playoff = build_knockout_stage(
        round_name="Third-place Playoff",
        teams=sf_result.eliminated,
        stage_type=StageType.PLAYOFF,
    )
    third_place_result, _ = resolve_stage(resolver, third_place_playoff)
    print_knockout_summary(third_place_result)

    final = build_knockout_stage(
        round_name="Final",
        teams=sf_result.qualifiers,
        stage_type=StageType.FINAL,
    )
    final_result, _ = resolve_stage(resolver, final)
    print_knockout_summary(final_result)

    print("Tournament Summary")
    print("------------------")
    print(f"Champion: {final_result.winner}")
    print(f"Runner-up: {final_result.runner_up}")
    print(f"Third place: {third_place_result.winner}")
    print(f"Fourth place: {third_place_result.runner_up}")


if __name__ == "__main__":
    main()