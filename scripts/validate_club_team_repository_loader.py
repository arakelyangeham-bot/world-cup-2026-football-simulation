#validate_club_team_repository_loader

from pathlib import Path

from scripts.club_team_repository_loader import (
    load_club_team_repository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "team_repositories"
    / "premier_league_validation_repository.csv"
)


def main() -> None:
    repository = load_club_team_repository(REPOSITORY_PATH)

    arsenal = repository["Arsenal"]

    assert "rating_prior" in arsenal
    assert "opta_rating" in arsenal
    assert "fifa_points" not in arsenal

    print("Club Repository Validation")
    print("==========================")
    print(f"Clubs loaded: {len(repository)}")
    print(f"Arsenal Opta rating: {arsenal['opta_rating']}")
    print(f"Arsenal rating prior: {arsenal['rating_prior']}")
    print()
    print("All club repository checks passed.")


if __name__ == "__main__":
    main()