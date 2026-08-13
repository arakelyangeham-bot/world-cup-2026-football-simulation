# test_probability_engine.py

from inference.probability_engine import ProbabilityEngine


def main():
    engine = ProbabilityEngine(mode="ml")

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

    probabilities = engine.predict_match(home_team, away_team)

    print(probabilities)
    print("Sum:", sum(probabilities.values()))


if __name__ == "__main__":
    main()