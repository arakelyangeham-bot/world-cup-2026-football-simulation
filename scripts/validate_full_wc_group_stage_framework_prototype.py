#validate_full_wc_group_stage_framework_prototype.py

from simulation.competition import (
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


def build_group_match_results(group_name: str, teams: list[str]) -> list[MatchResult]:
    """
    Build deterministic placeholder group-stage results.

    These results are not intended to be realistic. They only validate whether
    the full World Cup group-stage structure can be expressed through the
    generic competition framework.
    """

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
                    "study": "024",
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
            "study": "024",
            "prototype": "full_group_stage_framework",
        },
    )


def main() -> None:
    resolver = StageResolver()

    automatic_qualifiers: list[str] = []
    eliminated_teams: list[str] = []

    print("Framework-backed World Cup Group Stage Prototype")
    print("================================================")
    print()

    for group_name, teams in WORLD_CUP_GROUPS.items():
        stage = build_group_stage(group_name, teams)

        stage_result = resolver.resolve(stage)
        advancement_result = stage.advancement_rule.apply(stage_result)

        automatic_qualifiers.extend(advancement_result.qualifiers)
        eliminated_teams.extend(advancement_result.eliminated)

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

    print("Summary")
    print("-------")
    print(f"Groups resolved: {len(WORLD_CUP_GROUPS)}")
    print(f"Total group-stage matches: {len(WORLD_CUP_GROUPS) * 6}")
    print(f"Automatic qualifiers: {len(automatic_qualifiers)}")
    print(f"Eliminated by top-two rule: {len(eliminated_teams)}")
    print()

    print("Automatic Qualifiers")
    print("--------------------")
    for team in automatic_qualifiers:
        print(team)


if __name__ == "__main__":
    main()