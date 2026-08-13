#sofascore_discover_international_seasons.py

from time import sleep
import pandas as pd

from sofascore_utils import BASE_URL, OUT_DIR, get_json

REQUEST_DELAY = 3

IN_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_international_competitions.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_international_seasons.csv"

TARGET_YEARS = {"23/24", "24/25", "25/26", "2023", "2024", "2025", "2026"}


def get_competition_seasons(row):
    competition_id = row["unique_tournament_id"]

    url = f"{BASE_URL}/unique-tournament/{competition_id}/seasons"
    data = get_json(url)

    rows = []

    for season in data.get("seasons", []):
        season_year = season.get("year")

        if season_year not in TARGET_YEARS:
            continue

        rows.append({
            "competition": row["competition_name"],
            "competition_key": row["competition_key"],
            "competition_type": "international",
            "competition_id": competition_id,
            "season": season.get("name"),
            "season_year": season_year,
            "season_id": season.get("id"),
        })

    return rows


if __name__ == "__main__":
    competitions = pd.read_csv(IN_FILE)

    all_rows = []

    for idx, row in competitions.iterrows():
        print(
            f"[{idx+1}/{len(competitions)}] "
            f"Discovering seasons for {row['competition_name']}"
        )

        rows = get_competition_seasons(row)
        all_rows.extend(rows)

        sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["competition_id", "season_id"]
    )

    print(df)
    print(df.shape)

    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} international competition-season rows to {OUT_FILE}")