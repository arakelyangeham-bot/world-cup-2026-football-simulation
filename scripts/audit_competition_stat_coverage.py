#audit_competition_stat_coverage.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "competition_stat_coverage.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHECK_STATS = [
    "minutesPlayed",
    "rating",
    "goals",
    "assists",
    "keyPasses",
    "tackles",
    "interceptions",
    "clearances",
    "saves",
    "cleanSheet",
    "expectedGoals",
    "expectedAssists",
    "shotsOnTarget",
    "accurateCrosses",
    "successfulDribbles",
    "ballRecovery",
]

df = pd.read_csv(STATS_FILE, dtype={"season_year": str})

rows = []

for (competition, season_year), group in df.groupby(["competition", "season_year"], dropna=False):
    row = {
        "competition": competition,
        "season_year": season_year,
        "rows": len(group),
        "players": group["player_id"].nunique(),
    }

    for stat in CHECK_STATS:
        if stat in group.columns:
            row[f"{stat}_coverage"] = group[stat].notna().mean()
        else:
            row[f"{stat}_coverage"] = 0.0

    rows.append(row)

out = pd.DataFrame(rows).sort_values(
    ["rating_coverage", "rows"],
    ascending=[True, False],
)

out.to_csv(OUT_FILE, index=False)

print(out.head(50))
print(f"Wrote: {OUT_FILE}")