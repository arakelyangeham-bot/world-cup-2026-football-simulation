#audit_sofascore_seasons.py

from shared.competition_registry import get_competition
from shared.sofascore_season_loader import (
    load_sofascore_seasons,
)

seasons = load_sofascore_seasons()

def main() -> None:
    print(f"Registered Sofascore seasons: {len(seasons)}")
    print()

    print(
        f"{'Dataset ID':20}"
        f"{'Competition':25}"
        f"{'Year':>8}"
        f"{'Unique ID':>12}"
        f"{'Season ID':>12}"
    )
    print("-" * 77)

    for season in seasons:
        competition = get_competition(season.competition_key)

        print(
            f"{season.dataset_id:20}"
            f"{competition.display_name:25}"
            f"{season.year:8}"
            f"{season.unique_tournament_id:12}"
            f"{season.season_id:12}"
        )


if __name__ == "__main__":
    main()