#audit_lambda_override.py

from scripts.team_strength_loader import load_team_repository
from simulation.lambda_models import expected_goals


MATCHUPS = [
    ("Argentina", "France"),
    ("Spain", "Germany"),
    ("Brazil", "England"),
    ("Netherlands", "Portugal"),
    ("Canada", "Mexico"),
]


def main() -> None:
    repo = load_team_repository()

    print("Lambda Override Audit")
    print("---------------------")

    for team_a, team_b in MATCHUPS:
        a = repo[team_a]
        b = repo[team_b]

        heuristic = expected_goals(
            a,
            b,
            lambda_model="heuristic",
        )

        calibrated = expected_goals(
            a,
            b,
            lambda_model="calibrated",
        )

        print()
        print(f"{team_a} vs {team_b}")
        print("-" * (len(team_a) + len(team_b) + 4))
        print(f"heuristic : {heuristic[0]:.3f}, {heuristic[1]:.3f}")
        print(f"calibrated: {calibrated[0]:.3f}, {calibrated[1]:.3f}")
        print(
            f"delta     : "
            f"{calibrated[0] - heuristic[0]:+.3f}, "
            f"{calibrated[1] - heuristic[1]:+.3f}"
        )


if __name__ == "__main__":
    main()