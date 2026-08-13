# test_match_predictor.py

from inference.match_predictor import MatchPredictor


def main():
    predictor = MatchPredictor()

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

    probabilities = predictor.predict_match(home_team, away_team)

    print(probabilities)
    print("Sum:", sum(probabilities.values()))


if __name__ == "__main__":
    main()