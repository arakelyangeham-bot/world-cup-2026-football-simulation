#audit_historical_competitions.py

from pathlib import Path
import pandas as pd

from shared.historical_match_catalog import (
    get_available_historical_match_datasets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sofascore"

EXPECTED_MATCH_COUNTS = {
    ("world_cup", 2010): 64,
    ("world_cup", 2014): 64,
    ("world_cup", 2018): 64,
    ("world_cup", 2022): 64,
    ("euro", 2012): 31,
    ("euro", 2016): 51,
    ("euro", 2020): 51,
    ("euro", 2024): 51,
}


def audit_dataset(dataset) -> dict:
    path = RAW_DIR / dataset.filename
    expected = EXPECTED_MATCH_COUNTS.get(
        (dataset.competition_key, dataset.year)
    )

    if not path.exists():
        return {
            "competition": dataset.competition.display_name,
            "year": dataset.year,
            "source": dataset.source,
            "filename": dataset.filename,
            "expected": expected,
            "actual": None,
            "columns": None,
            "status": "MISSING",
        }

    df = pd.read_csv(path)
    actual = len(df)

    required_columns = {
        "event_id",
        "date",
        "stage",
        "round",
        "round_number",
        "home_team",
        "home_team_id",
        "away_team",
        "away_team_id",
        "home_score",
        "away_score",
        "status_code",
        "status_desc",
        "winner",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        status = "BAD_SCHEMA"
    elif expected is not None and actual != expected:
        status = "BAD_COUNT"
    else:
        status = "OK"

    return {
        "competition": dataset.competition.display_name,
        "year": dataset.year,
        "source": dataset.source,
        "filename": dataset.filename,
        "expected": expected,
        "actual": actual,
        "columns": len(df.columns),
        "status": status,
    }


def main() -> None:
    datasets = get_available_historical_match_datasets()
    rows = [audit_dataset(dataset) for dataset in datasets]

    print(f"Registered datasets: {len(rows)}")
    print()

    print(
        f"{'Competition':30}"
        f"{'Year':>8}"
        f"{'Expected':>10}"
        f"{'Actual':>10}"
        f"{'Cols':>8}"
        f"{'Status':>14}"
        f"  Filename"
    )
    print("-" * 105)

    for row in rows:
        expected = "" if row["expected"] is None else str(row["expected"])
        actual = "" if row["actual"] is None else str(row["actual"])
        columns = "" if row["columns"] is None else str(row["columns"])

        print(
            f"{row['competition'][:30]:30}"
            f"{row['year']:8}"
            f"{expected:>10}"
            f"{actual:>10}"
            f"{columns:>8}"
            f"{row['status']:>14}"
            f"  {row['filename']}"
        )

    bad_rows = [row for row in rows if row["status"] != "OK"]

    print()
    print("Summary")
    print("-------")
    print(f"OK: {len(rows) - len(bad_rows)}")
    print(f"Needs attention: {len(bad_rows)}")


if __name__ == "__main__":
    main()