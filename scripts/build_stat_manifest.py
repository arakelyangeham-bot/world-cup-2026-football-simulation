#build_stat_manifest.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"
OUT_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "stat_manifest.csv"

IDENTITY_COLUMNS = {
    "player_id",
    "player",
    "player_slug",
}

INTERNAL_COLUMNS = {
    "id",
    "type",
    "totalRating",
    "countRating",
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "team_id",
    "team",
    "team_slug",
}

df = pd.read_csv(
    IN_FILE,
    dtype={"season_year": str},
)
rows = []

for col in df.columns:
    if col in IDENTITY_COLUMNS:
        aggregation = "identity"
        weight_by = ""
    elif col == "minutesPlayed":
        aggregation = "sum"
        weight_by = ""
    elif col.lower().endswith("percentage"):
        aggregation = "weighted_mean"
        weight_by = "minutesPlayed"
    elif col in {"rating"}:
        aggregation = "weighted_mean"
        weight_by = "minutesPlayed"
    else:
        aggregation = "sum"
        weight_by = ""
    

    include_in_model = col not in INTERNAL_COLUMNS

    rows.append({
        "stat": col,
        "aggregation": aggregation,
        "weight_by": weight_by,
        "enabled": True,
        "include_in_model": include_in_model,
    })

manifest = pd.DataFrame(rows)

manifest.to_csv(OUT_FILE, index=False)

print(manifest)
print(f"Saved {len(manifest)} rows to {OUT_FILE}")