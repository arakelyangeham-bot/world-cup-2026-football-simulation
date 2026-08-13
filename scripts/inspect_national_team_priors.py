#inspect_national_team_priors.py

from shared.national_team_priors import load_national_team_priors


def main():
    priors = load_national_team_priors()

    print("Teams loaded:", len(priors))
    print()

    for team, values in list(priors.items())[:20]:
        print(team, values)


if __name__ == "__main__":
    main()