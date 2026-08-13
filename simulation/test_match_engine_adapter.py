#test_match_engine_adapter.py

from simulation.match_engine_adapter import simulate_match_score


def main():
    team1_strength = {
        "attack": 88,
        "midfield": 86,
        "defense": 84,
        "gk": 85,
    }

    team2_strength = {
        "attack": 82,
        "midfield": 80,
        "defense": 81,
        "gk": 83,
    }

    print("Poisson:")
    print(simulate_match_score(team1_strength, team2_strength, mode="poisson"))

    print()
    print("ML-guided:")
    print(simulate_match_score(team1_strength, team2_strength, mode="ml"))


if __name__ == "__main__":
    main()