# audit_wc2026_structure.py

from wc2026_data import GROUPS


def main():
    print("Groups:", len(GROUPS))

    total_teams = sum(len(teams) for teams in GROUPS.values())
    print("Total teams:", total_teams)

    print()
    print("Group sizes:")
    for group, teams in sorted(GROUPS.items()):
        print(group, len(teams), teams)

    all_teams = [
        team
        for teams in GROUPS.values()
        for team in teams
    ]

    duplicates = sorted({
        team
        for team in all_teams
        if all_teams.count(team) > 1
    })

    print()
    print("Duplicate teams:", duplicates)

    if len(GROUPS) != 12:
        raise ValueError("Expected 12 groups")

    if total_teams != 48:
        raise ValueError("Expected 48 teams")

    if duplicates:
        raise ValueError("Duplicate teams found")

    print()
    print("World Cup 2026 group structure audit passed.")


if __name__ == "__main__":
    main()