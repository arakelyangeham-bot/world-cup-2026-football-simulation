#sofascore_build_competitions_config.py

import pandas as pd

from sofascore_utils import OUT_DIR

IN_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_league_seasons.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_competitions.csv"

df = pd.read_csv(IN_FILE)

competitions = pd.DataFrame({
    "competition": df["league"],
    "competition_type": "club_league",
    "competition_id": df["league_id"],
    "season_id": df["season_id"],
    "season_year": df["season_year"],
})

competitions = competitions.drop_duplicates(
    subset=["competition_id", "season_id"]
)

print(competitions)

competitions.to_csv(OUT_FILE, index=False)

print(f"Saved {len(competitions)} competitions to {OUT_FILE}")
