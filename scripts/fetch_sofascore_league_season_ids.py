from __future__ import annotations
import re
from pathlib import Path
from time import sleep

import pandas as pd

from sofascore_utils import BASE_URL, OUT_DIR, get_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_DIR = PROJECT_ROOT / "data" / "roster"
ROSTER_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = ROSTER_DIR / "league_and_season_ids.csv"
OUTPUT_FILE = ROSTER_DIR / "league_and_season_ids_filled.csv"
FAILED_FILE = OUT_DIR / "league_and_season_ids_failed.csv"

TARGET_YEARS = [2023, 2024, 2025]
REQUEST_DELAY = 10
CHECKPOINT_EVERY = 1


def is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def normalize_year_key(value: object) -> str:
    return str(value).strip()


def season_id_map(league_id: int) -> tuple[dict[str, int], str]:
    """Return many possible season-year labels mapped to Sofascore season IDs."""
    url = f"{BASE_URL}/unique-tournament/{league_id}/seasons"
    data = get_json(url)

    seasons: dict[str, int] = {}
    for season in data.get("seasons", []):
        sid = season.get("id")
        if sid is None:
            continue

        year = normalize_year_key(season.get("year", ""))
        name = normalize_year_key(season.get("name", ""))
        text = f"{year} {name}"

        if year:
            seasons[year] = int(sid)

        # Examples this catches: 2025, 25/26, 2025/26, 2025/2026.
        for key in re.findall(r"20\d{2}|\d{2}/\d{2}|\d{4}/\d{2}|\d{4}/\d{4}", text):
            seasons[key] = int(sid)

    return seasons, url


def pick_season_id(seasons: dict[str, int], target_year: int) -> int | None:
    """
    Pick the season ID for a target data year.

    For European fall-spring leagues, 2025 means 2025/26 or 25/26.
    For calendar-year leagues, 2025 means 2025.
    """
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
    pd.DataFrame(failed_rows).drop_duplicates(subset=["league_id"]).to_csv(FAILED_FILE, index=False)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"league_name", "league_id"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Input file is missing required columns: {sorted(missing_required)}")

    for year in TARGET_YEARS:
        col = f"season_id_{year}"
        if col not in df.columns:
            df[col] = ""

    for col in ["lookup_status", "season_source_url", "available_season_keys"]:
        if col not in df.columns:
            df[col] = ""

    failed_rows: list[dict] = []

    for idx, row in df.iterrows():
        league_name = str(row["league_name"])

        if is_blank(row.get("league_id")):
            df.at[idx, "lookup_status"] = "missing_league_id"
            failed_rows.append({
                "league_name": league_name,
                "league_id": "",
                "error": "missing_league_id",
            })
            continue

        league_id = int(float(row["league_id"]))
        print(f"[{idx + 1}/{len(df)}] Fetching seasons for {league_name} ({league_id})")

        try:
            seasons, source_url = season_id_map(league_id)
            df.at[idx, "season_source_url"] = source_url
            df.at[idx, "available_season_keys"] = ";".join(sorted(seasons.keys()))

            missing_years: list[str] = []
            for year in TARGET_YEARS:
                col = f"season_id_{year}"
                sid = pick_season_id(seasons, year)
                if sid is not None:
                    df.at[idx, col] = sid
                elif is_blank(df.at[idx, col]):
                    missing_years.append(str(year))

            if missing_years:
                df.at[idx, "lookup_status"] = "missing_seasons_" + "_".join(missing_years)
            else:
                df.at[idx, "lookup_status"] = "confirmed"

        except Exception as exc:
            error_text = str(exc)
            print(f"FAILED: {league_name} ({league_id}) -> {error_text}")
            df.at[idx, "lookup_status"] = "failed"
            failed_rows.append({
                "league_name": league_name,
                "league_id": league_id,
                "error": error_text,
            })

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_outputs(df, failed_rows)
            print(f"Checkpoint saved: {OUTPUT_FILE}")

        sleep(REQUEST_DELAY)

    save_outputs(df, failed_rows)
    print("Done.")
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Failures: {FAILED_FILE}")
    print(df["lookup_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
