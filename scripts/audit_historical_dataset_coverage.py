#audit_historical_dataset_coverage.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print_section("Competition coverage")
    coverage = (
        df.groupby(["competition", "competition_key"])
        .agg(
            matches=("event_id", "count"),
            years=("year", lambda s: ", ".join(map(str, sorted(s.unique())))),
            teams=("home_team", "nunique"),
        )
        .reset_index()
        .sort_values("matches", ascending=False)
    )

    for _, row in coverage.iterrows():
        print(
            f"{row['competition']:<35} "
            f"{row['matches']:>5} matches  "
            f"years: {row['years']}"
        )

    print_section("Year coverage")
    year_counts = df["year"].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"{year}: {count}")

    print_section("Result distribution")
    result_counts = df["result"].value_counts()
    result_rates = df["result"].value_counts(normalize=True)

    for result, count in result_counts.items():
        print(f"{result:<10} {count:>5}  {result_rates[result]:.3f}")

    print_section("Team coverage")
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    print(f"Unique teams: {len(teams)}")
    print("Sample:", ", ".join(teams[:20]))

    print_section("Date coverage")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print(f"Earliest: {dates.min()}")
        print(f"Latest:   {dates.max()}")

    print_section("Dataset IDs")
    if "dataset_id" in df.columns:
        dataset_counts = df["dataset_id"].value_counts().sort_index()
        for dataset_id, count in dataset_counts.items():
            print(f"{dataset_id:<20} {count}")


if __name__ == "__main__":
    main()