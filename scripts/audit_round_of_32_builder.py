# audit_round_of_32_builder.py

from scripts.wc2026_group_stage import (
    simulate_group_stage,
    extract_qualifiers_from_standings,
)
from scripts.team_strength_loader import load_team_repository
from scripts.wc2026_knockout_mapping import build_round_of_32


def main():
    team_repository = load_team_repository()

    standings = simulate_group_stage(team_repository)
    qualifiers = extract_qualifiers_from_standings(standings)

    matches = build_round_of_32(qualifiers)

    print(f"Round of 32 matches: {len(matches)}")
    print()

    for match in matches:
        print(
            f"M{match.match_number}: "
            f"{match.team1.team} ({match.team1.finish}{match.team1.group})"
            f" vs "
            f"{match.team2.team} ({match.team2.finish}{match.team2.group})"
        )
    
    print()

    used = []

    for match in matches:
        used.append(match.team1.team)
        used.append(match.team2.team)

    print("Unique teams:", len(set(used)))
    print("Total slots :", len(used))

    assert len(used) == 32
    assert len(set(used)) == 32

    print("Round of 32 uniqueness audit passed.")


if __name__ == "__main__":
    main()