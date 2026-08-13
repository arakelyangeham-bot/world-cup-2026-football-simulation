#smoke_test_production_prediction.py

from inference.match_predictor import MatchPredictor
from team_strength_loader import load_complete_team_strengths


def main():
    teams = load_complete_team_strengths()

    home = "France"
    away = "Brazil"

    predictor = MatchPredictor()

    probabilities = predictor.predict_match(
        teams[home],
        teams[away],
    )

    print(f"{home} vs {away}")
    print(probabilities)

    print()
    print(f"Probability sum: {sum(probabilities.values()):.6f}")


if __name__ == "__main__":
    main()