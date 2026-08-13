# inspect_fifa_points_coverage.py

from shared.national_team_priors import load_national_team_priors
from wc2026_data import GROUPS


def main():
    priors = load_national_team_priors()
    teams = sorted(team for group in GROUPS.values() for team in group)

    missing = []

    for team in teams:
        values = priors.get(team, {})
        if "fifa_points" not in values:
            missing.append(team)

    print("World Cup teams:", len(teams))
    print("Teams with FIFA points:", len(teams) - len(missing))
    print("Missing:", len(missing))

    print("\n--- GROUPS entry ---")
    for team in missing:
        print(repr(team))

    print("\n--- Matching keys in priors ---")
    for team in sorted(priors):
        if "United" in team or "USA" in team:
            print(repr(team))

    print("\n--- All keys starting with U ---")
    for team in sorted(priors):
        if team.startswith("U"):
            print(repr(team))


if __name__ == "__main__":
    main()