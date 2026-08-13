# audit_full_knockout.py

from team_strength_loader import load_poisson_team_strengths
from wc2026_group_stage import simulate_group_stage, extract_qualifiers_from_standings
from wc2026_knockout_mapping import build_round_of_32
from wc2026_knockout_stage import (
    simulate_knockout_match,
    build_next_round,
    build_third_place_playoff,
    ROUND_OF_16_PAIRINGS,
    QUARTERFINAL_PAIRINGS,
    SEMIFINAL_PAIRINGS,
    FINAL_PAIRING,
)


def simulate_round(matches, strengths):
    return [
        simulate_knockout_match(match, strengths)
        for match in matches
    ]


def print_round(name, results):
    print()
    print(name)

    for result in results:
        print(
            f"M{result.match_number}: "
            f"{result.team1.team} {result.goals_team1}-"
            f"{result.goals_team2} {result.team2.team} "
            f"-> {result.winner.team}"
        )


def main():
    strengths = load_poisson_team_strengths()

    standings = simulate_group_stage(strengths)
    qualifiers = extract_qualifiers_from_standings(standings)

    round_of_32 = build_round_of_32(qualifiers)
    r32_results = simulate_round(round_of_32, strengths)

    round_of_16 = build_next_round(r32_results, ROUND_OF_16_PAIRINGS, 89)
    r16_results = simulate_round(round_of_16, strengths)

    quarterfinals = build_next_round(r16_results, QUARTERFINAL_PAIRINGS, 97)
    qf_results = simulate_round(quarterfinals, strengths)

    semifinals = build_next_round(qf_results, SEMIFINAL_PAIRINGS, 101)
    sf_results = simulate_round(semifinals, strengths)

    third_place = build_third_place_playoff(sf_results)
    third_place_result = simulate_round(third_place, strengths)

    final = build_next_round(sf_results, FINAL_PAIRING, 104)
    final_result = simulate_round(final, strengths)

    print_round("Round of 32", r32_results)
    print_round("Round of 16", r16_results)
    print_round("Quarterfinals", qf_results)
    print_round("Semifinals", sf_results)
    print_round("Third-place playoff", third_place_result)
    print_round("Final", final_result)

    champion = final_result[0].winner.team

    print()
    print("Champion:", champion)

    assert len(r32_results) == 16
    assert len(r16_results) == 8
    assert len(qf_results) == 4
    assert len(sf_results) == 2
    assert len(third_place_result) == 1
    assert len(final_result) == 1

    print()
    print("Full knockout audit passed.")


if __name__ == "__main__":
    main()