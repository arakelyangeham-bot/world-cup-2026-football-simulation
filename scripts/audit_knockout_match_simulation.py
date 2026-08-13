# audit_knockout_match_simulation.py

from team_strength_loader import load_poisson_team_strengths
from wc2026_group_stage import simulate_group_stage, extract_qualifiers_from_standings
from wc2026_knockout_mapping import build_round_of_32
from wc2026_knockout_stage import simulate_knockout_match


def main():
    strengths = load_poisson_team_strengths()

    standings = simulate_group_stage(strengths)
    qualifiers = extract_qualifiers_from_standings(standings)
    matches = build_round_of_32(qualifiers)

    results = [
        simulate_knockout_match(match, strengths)
        for match in matches
    ]

    print("Round of 32 results:", len(results))
    print()

    for result in results:
        flags = []
        if result.went_to_extra_time:
            flags.append("ET")
        if result.went_to_penalties:
            flags.append("PEN")

        suffix = f" ({', '.join(flags)})" if flags else ""

        print(
            f"M{result.match_number}: "
            f"{result.team1.team} {result.goals_team1}-"
            f"{result.goals_team2} {result.team2.team} "
            f"-> {result.winner.team}{suffix}"
        )

    winners = [result.winner.team for result in results]

    print()
    print("Winners:", len(winners))
    print("Unique winners:", len(set(winners)))

    assert len(winners) == 16
    assert len(set(winners)) == 16

    print("Knockout match simulation audit passed.")


if __name__ == "__main__":
    main()