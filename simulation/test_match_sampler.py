# test_match_sampler.py

from simulation.match_sampler import MatchSampler


def main():
    sampler = MatchSampler(mode="ml")

    home_team = {
        "attack": 88,
        "midfield": 86,
        "defense": 84,
        "gk": 85,
    }

    away_team = {
        "attack": 82,
        "midfield": 80,
        "defense": 81,
        "gk": 83,
    }

    print("Group-style outcome:")
    print(sampler.sample_outcome(home_team, away_team))

    print()
    print("Knockout-style winner:")
    print(sampler.sample_knockout_winner(home_team, away_team))


if __name__ == "__main__":
    main()