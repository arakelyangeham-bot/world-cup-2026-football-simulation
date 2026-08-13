#validate_rating_prior_compatibility

from scripts.team_strength_loader import load_team_repository
from simulation.match_engine_adapter import (
    repository_entry_to_poisson_features,
    simulate_match_score,
)


def main() -> None:
    repository = load_team_repository()

    france = repository["France"]
    argentina = repository["Argentina"]

    assert "rating_prior" in france
    assert "fifa_points" in france
    assert france["rating_prior"] == france["fifa_points"]

    france_features = repository_entry_to_poisson_features(france)

    assert "rating_prior" in france_features
    assert "fifa_points" in france_features
    assert (
        france_features["rating_prior"]
        == france_features["fifa_points"]
    )

    goals_france, goals_argentina = simulate_match_score(
        france,
        argentina,
    )

    print("Rating Prior Compatibility Validation")
    print("=====================================")
    print(f"France rating_prior: {france['rating_prior']}")
    print(f"France fifa_points alias: {france['fifa_points']}")
    print(
        "Converted feature prior: "
        f"{france_features['rating_prior']}"
    )
    print(
        "Sample match: "
        f"France {goals_france}-{goals_argentina} Argentina"
    )
    print()
    print("All compatibility checks passed.")


if __name__ == "__main__":
    main()