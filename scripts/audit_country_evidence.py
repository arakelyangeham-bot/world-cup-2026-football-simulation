#audit_country_evidence.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"
PROFILES_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_profiles.csv"
COMPETITION_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "competition_manifest.csv"

OUT_FILE = PROJECT_ROOT / "outputs" / "papua_new_guinea_evidence.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET_COUNTRY = "Papua New Guinea"

stats = pd.read_csv(STATS_FILE, dtype={"season_year": str})
profiles = pd.read_csv(PROFILES_FILE)
competitions = pd.read_csv(COMPETITION_FILE, dtype={"season_year": str})

df = stats.merge(
    profiles[["player_id", "country", "position", "positions_detailed"]],
    on="player_id",
    how="left",
)

df = df.merge(
    competitions[
        [
            "competition_id",
            "season_id",
            "recency_weight",
            "competition_importance",
        ]
    ],
    on=["competition_id", "season_id"],
    how="left",
)

df["row_weight"] = (
    pd.to_numeric(df["minutesPlayed"], errors="coerce").fillna(0)
    * pd.to_numeric(df["recency_weight"], errors="coerce").fillna(1)
    * pd.to_numeric(df["competition_importance"], errors="coerce").fillna(1)
)

country_df = df[df["country"] == TARGET_COUNTRY].copy()

cols = [
    "player_id",
    "player",
    "country",
    "position",
    "positions_detailed",
    "competition",
    "competition_type",
    "season_year",
    "team",
    "minutesPlayed",
    "row_weight",
    "rating",
    "goals",
    "assists",
    "keyPasses",
    "tackles",
    "interceptions",
    "expectedGoals",
    "expectedAssists",
]

cols = [c for c in cols if c in country_df.columns]

print(country_df[cols].sort_values(["player", "competition"]).head(100))
print("\nRows:", len(country_df))
print("Players:", country_df["player_id"].nunique())
print("Non-null ratings:", country_df["rating"].notna().sum())
print("Total minutes:", pd.to_numeric(country_df["minutesPlayed"], errors="coerce").sum())

country_df[cols].to_csv(OUT_FILE, index=False)

print(f"Wrote: {OUT_FILE}")