# test_probability_adapter.py

from simulation.probability_adapter import ProbabilityAdapter


def main():
    adapter = ProbabilityAdapter(mode="ml")

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

    probabilities = adapter.predict_match_probabilities(
        home_team,
        away_team,
    )

    print(probabilities)
    print("Sum:", sum(probabilities.values()))


if __name__ == "__main__":
    main()