#audit_player_identity.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

df = pd.read_csv(FILE, dtype={"season_year": str})

print(f"Rows: {len(df):,}")
print(f"Unique player_id: {df['player_id'].nunique():,}")
print(f"Unique player names: {df['player'].nunique():,}")

duplicates = (
    df.groupby("player")["player_id"]
      .nunique()
      .sort_values(ascending=False)
)

print("\nPlayers with multiple IDs:")
print(duplicates[duplicates > 1].head(50))