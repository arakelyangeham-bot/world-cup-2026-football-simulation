#sofascore_discover_league_seasons.py

from time import sleep
import pandas as pd

from sofascore_utils import BASE_URL, OUT_DIR, get_json

REQUEST_DELAY = 3

LEAGUES = [
    {"league": "Premier League", "league_id": 17},
]

TARGET_YEARS = {"23/24", "24/25", "25/26"}

OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_league_seasons.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_league_seasons(league):
    url = f"{BASE_URL}/unique-tournament/{league['league_id']}/seasons"
    data = get_json(url)

    rows = []

    for season in data.get("seasons", []):
        if season.get("year") not in TARGET_YEARS:
            continue

        rows.append({
            "league": league["league"],
            "league_id": league["league_id"],
            "season": season.get("name"),
            "season_year": season.get("year"),
            "season_id": season.get("id"),
        })

    return rows


if __name__ == "__main__":
    all_rows = []

    for league in LEAGUES:
        print(f"Discovering seasons for {league['league']}")

        rows = get_league_seasons(league)
        all_rows.extend(rows)

        sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows).sort_values(
        ["league", "season_year"]
    )

    print(df)

    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} league-season rows to {OUT_FILE}")