#audit_player_evidence_rows.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"

df = pd.read_csv(FILE, dtype={"season_year": str})

print(f"Rows: {len(df):,}")
print(f"Unique player_id: {df['player_id'].nunique():,}")
print(f"Unique players: {df['player'].nunique():,}")

rows_per_player = (
    df.groupby("player_id")
      .size()
      .sort_values(ascending=False)
)

print("\nRows per player_id:")
print(rows_per_player.describe())

print("\nTop repeated player_ids:")
print(rows_per_player.head(25))

print("\nSample repeated evidence rows:")
repeated_ids = rows_per_player[rows_per_player > 1].head(10).index

cols = [
    "player_id",
    "player",
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "team",
    "minutesPlayed",
    "rating",
]

cols = [c for c in cols if c in df.columns]

print(
    df[df["player_id"].isin(repeated_ids)]
    .sort_values(["player_id", "competition", "season_year"])
    [cols]
    .head(100)
)