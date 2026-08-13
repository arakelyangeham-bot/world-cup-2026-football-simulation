#audit_match_engine_modes.py

import random
import numpy as np

import simulation.simulation_config as simulation_config

from scripts.team_strength_loader import load_team_repository
from simulation.lambda_models import expected_goals
from simulation.match_engine_adapter import simulate_match_score
from simulation.match_engine_adapter import get_cached_probabilities


MATCHUPS = [
    ("Argentina", "France"),
    ("Spain", "Germany"),
    ("Brazil", "England"),
]


def run_once(team_a, team_b, mode: str, lambda_model: str):
    simulation_config.LAMBDA_MODEL = lambda_model

    lambdas = expected_goals(
        team_a,
        team_b,
        lambda_model=lambda_model,
    )

    score = simulate_match_score(
        team_a,
        team_b,
        mode=mode,
    )

    probabilities = None

    if mode == "ml":
        probabilities = get_cached_probabilities(
            team_a,
            team_b,
        )

    return lambdas, score, probabilities


def main() -> None:
    repo = load_team_repository()

    random.seed(42)
    np.random.seed(42)

    original_lambda_model = simulation_config.LAMBDA_MODEL

    try:
        print("Match Engine Mode Audit")
        print("-----------------------")

        for team_a_name, team_b_name in MATCHUPS:
            team_a = repo[team_a_name]
            team_b = repo[team_b_name]

            print()
            print(f"{team_a_name} vs {team_b_name}")
            print("-" * (len(team_a_name) + len(team_b_name) + 4))

            for mode in ["poisson", "ml"]:
                for lambda_model in ["heuristic", "calibrated"]:
                    lambdas, score, probabilities = run_once(
                        team_a,
                        team_b,
                        mode=mode,
                        lambda_model=lambda_model,
                    )

                    print()
                    print(f"mode={mode} lambda_model={lambda_model}")
                    print(f"lambda: {lambdas[0]:.3f}, {lambdas[1]:.3f}")
                    print(f"score : {score[0]}-{score[1]}")

                    if probabilities is not None:
                        print(
                            "ml probabilities: "
                            f"home={probabilities['home_win']:.3f}, "
                            f"draw={probabilities['draw']:.3f}, "
                            f"away={probabilities['away_win']:.3f}"
                        )

    finally:
        simulation_config.LAMBDA_MODEL = original_lambda_model


if __name__ == "__main__":
    main()