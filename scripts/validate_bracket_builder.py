#validate_bracket_builder.py

from simulation.competition import BracketBuilder


def main() -> None:
    teams = [
        "Argentina",
        "Brazil",
        "Morocco",
        "Japan",
    ]

    builder = BracketBuilder()
    bracket = builder.build_high_low_bracket(
        teams=teams,
        name="Example Semifinal Bracket",
    )

    print(f"Bracket: {bracket.name}")
    print(f"Builder: {bracket.metadata['builder']}")
    print(f"Pairing type: {bracket.metadata['pairing_type']}")
    print(f"Teams: {bracket.metadata['team_count']}")
    print(f"Ties: {bracket.metadata['tie_count']}")
    print()

    for index, tie in enumerate(bracket.ties, start=1):
        print(
            f"Tie {index}: "
            f"{tie.team1} vs {tie.team2} "
            f"(seeds {tie.metadata['seed_team1']} vs {tie.metadata['seed_team2']})"
        )


if __name__ == "__main__":
    main()