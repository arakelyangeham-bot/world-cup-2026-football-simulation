#audit_historical_match_catalog.py

from shared.historical_match_catalog import (
    get_available_historical_match_datasets,
)


def main() -> None:
    datasets = get_available_historical_match_datasets()

    print(f"Registered historical match datasets: {len(datasets)}")
    print()

    print(
        f"{'Competition':25}"
        f"{'Year':>8}"
        f"{'Source':>14}"
        f"  Filename"
    )
    print("-" * 85)

    for dataset in datasets:
        print(
            f"{dataset.competition.display_name:25}"
            f"{dataset.year:8}"
            f"{dataset.source:>14}"
            f"  {dataset.filename}"
        )


if __name__ == "__main__":
    main()