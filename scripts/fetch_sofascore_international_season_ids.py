#fetch_sofascore_international_season_ids.py

from __future__ import annotations

import re
from pathlib import Path
from time import sleep

import pandas as pd

from scripts.sofascore_utils import BASE_URL, get_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = PROJECT_ROOT / "outputs" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = METADATA_DIR / "sofascore_international_competitions.csv"
OUTPUT_FILE = METADATA_DIR / "sofascore_international_seasons_filled.csv"
FAILED_FILE = OUT_DIR / "sofascore_international_seasons_failed.csv"

TARGET_YEARS = [
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]

REQUEST_DELAY = 5


def is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def normalize_year_key(value: object) -> str:
    return str(value).strip()


def season_id_map(unique_tournament_id: int) -> tuple[dict[str, int], str]:
    url = f"{BASE_URL}/unique-tournament/{unique_tournament_id}/seasons"
    data = get_json(url)

    seasons: dict[str, int] = {}

    for season in data.get("seasons", []):
        season_id = season.get("id")
        if season_id is None:
            continue

        year = normalize_year_key(season.get("year", ""))
        name = normalize_year_key(season.get("name", ""))

        text = f"{year} {name}"

        if year:
            seasons[year] = int(season_id)

        for key in re.findall(
            r"20\d{2}|\d{2}/\d{2}|\d{4}/\d{2}|\d{4}/\d{4}",
            text,
        ):
            seasons[key] = int(season_id)

    return seasons, url


def pick_season_id(seasons: dict[str, int], target_year: int) -> int | None:
    candidates = [
        str(target_year),
        f"{str(target_year)[-2:]}/{str(target_year + 1)[-2:]}",
        f"{target_year}/{str(target_year + 1)[-2:]}",
        f"{target_year}/{target_year + 1}",
    ]

    for key in candidates:
        if key in seasons:
            return seasons[key]

    return None


def save_outputs(df: pd.DataFrame, failed_rows: list[dict]) -> None:
    df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(failed_rows).drop_duplicates(
        subset=["competition_key", "unique_tournament_id"],
    ).to_csv(FAILED_FILE, index=False)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}\n"
            "Create it with columns: "
            "competition_key,competition_name,unique_tournament_id"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "competition_key",
        "competition_name",
        "unique_tournament_id",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing columns: {sorted(missing)}")

    for year in TARGET_YEARS:
        column = f"season_id_{year}"
        if column not in df.columns:
            df[column] = pd.NA

        df[column] = df[column].astype("object")

    for column in [
        "lookup_status",
        "lookup_error",
        "season_source_url",
        "available_season_keys",
    ]:
        if column not in df.columns:
            df[column] = ""

    failed_rows: list[dict] = []

    for idx, row in df.iterrows():
        competition_key = str(row["competition_key"])
        competition_name = str(row["competition_name"])
        df.at[idx, "lookup_error"] = ""

        if is_blank(row.get("unique_tournament_id")):
            df.at[idx, "lookup_status"] = "missing_unique_tournament_id"
            failed_rows.append(
                {
                    "competition_key": competition_key,
                    "competition_name": competition_name,
                    "unique_tournament_id": "",
                    "error": "missing_unique_tournament_id",
                }
            )
            continue

        unique_tournament_id = int(float(row["unique_tournament_id"]))

        print(
            f"[{idx + 1}/{len(df)}] "
            f"Fetching seasons for {competition_name} "
            f"({unique_tournament_id})"
        )

        try:
            seasons, source_url = season_id_map(unique_tournament_id)
            print("Discovered seasons:")
            for key, value in sorted(seasons.items()):
                print(f"  {key} -> {value}")

            df.at[idx, "season_source_url"] = source_url
            df.at[idx, "available_season_keys"] = ";".join(
                sorted(seasons.keys())
            )

            missing_years: list[str] = []

            for year in TARGET_YEARS:
                column = f"season_id_{year}"
                season_id = pick_season_id(seasons, year)

                if season_id is not None:
                    df.at[idx, column] = season_id
                elif is_blank(df.at[idx, column]):
                    missing_years.append(str(year))

            if missing_years:
                df.at[idx, "lookup_status"] = (
                    "missing_seasons_" + "_".join(missing_years)
                )
            else:
                df.at[idx, "lookup_status"] = "confirmed"

        except Exception as exc:
            error_text = str(exc)
            df.at[idx, "lookup_error"] = error_text
            print(
                f"FAILED: {competition_name} "
                f"({unique_tournament_id}) -> {error_text}"
            )

            df.at[idx, "lookup_status"] = "failed"

            failed_rows.append(
                {
                    "competition_key": competition_key,
                    "competition_name": competition_name,
                    "unique_tournament_id": unique_tournament_id,
                    "error": error_text,
                }
            )

        save_outputs(df, failed_rows)
        print(f"Checkpoint saved: {OUTPUT_FILE}")

        sleep(REQUEST_DELAY)

    save_outputs(df, failed_rows)

    print()
    print("Done.")
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Failures: {FAILED_FILE}")
    print()
    print(df["lookup_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()