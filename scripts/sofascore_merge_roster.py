from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROSTER_FILE = PROJECT_ROOT / "data" / "roster" / "world_cup_2026_roster_with_sofascore_ids.csv"
STATS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_wc_player_stats.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_player_dataset.csv"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

roster = pd.read_csv(ROSTER_FILE)
stats = pd.read_csv(STATS_FILE)

roster = roster.drop(columns=["Unnamed: 7"], errors="ignore")

roster["sofascore_player_id"] = roster["sofascore_player_id"].astype("Int64")
stats["player_id"] = stats["player_id"].astype("Int64")

merged = roster.merge(
    stats,
    left_on="sofascore_player_id",
    right_on="player_id",
    how="left",
    suffixes=("", "_stats"),
)

merged["has_wc_stats"] = merged["minutesPlayed"].notna()

merged.to_csv(OUT_FILE, index=False)

print("Done.")
print(f"Roster rows: {len(roster)}")
print(f"Stats rows: {len(stats)}")
print(f"Merged rows: {len(merged)}")
print(f"Players with WC stats: {merged['has_wc_stats'].sum()}")
print(f"Wrote: {OUT_FILE}")