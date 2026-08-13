# audit_round_of_16_builder.py

from team_strength_loader import load_poisson_team_strengths
from wc2026_group_stage import simulate_group_stage, extract_qualifiers_from_standings
from wc2026_knockout_mapping import build_round_of_32
from wc2026_knockout_stage import (
    simulate_knockout_match,
    build_next_round,
    ROUND_OF_16_PAIRINGS,
)


def main():
    strengths = load_poisson_team_strengths()

    standings = simulate_group_stage(strengths)
    qualifiers = extract_qualifiers_from_standings(standings)
    round_of_32 = build_round_of_32(qualifiers)

    r32_results = [
        simulate_knockout_match(match, strengths)
        for match in round_of_32
    ]

    round_of_16 = build_next_round(
        r32_results,
        ROUND_OF_16_PAIRINGS,
        starting_match_number=89,
    )

    print("Round of 16 matches:", len(round_of_16))
    print()

    for match in round_of_16:
        print(
            f"M{match.match_number}: "
            f"{match.team1.team} vs {match.team2.team}"
        )

    teams = [
        team
        for match in round_of_16
        for team in (match.team1.team, match.team2.team)
    ]

    assert len(round_of_16) == 8
    assert len(teams) == 16
    assert len(set(teams)) == 16

    print()
    print("Round of 16 builder audit passed.")


if __name__ == "__main__":
    main()