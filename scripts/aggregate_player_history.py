#aggregate_player_history.py

from pathlib import Path
import pandas as pd
import numpy as np
import ast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPETITION_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "competition_manifest.csv"
PROFILES_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_profiles.csv"
STATS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"
STAT_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "stat_manifest.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_player_dataset.csv"

ROLE_PRIORITY = [
    "GK",
    "ST",
    "W",
    "AM",
    "CM",
    "DM",
    "FB",
    "CB",
    "WM",
]

SOFASCORE_ROLE_MAP = {
    "GK": "GK",
    "DC": "CB",
    "DL": "FB",
    "DR": "FB",
    "DM": "DM",
    "MC": "CM",
    "AM": "AM",
    "ML": "WM",
    "MR": "WM",
    "LW": "W",
    "RW": "W",
    "ST": "ST",
}

def parse_positions(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return [str(value)]

def infer_eligible_roles(positions_detailed):
    raw_positions = parse_positions(positions_detailed)

    roles = []
    for pos in raw_positions:
        mapped = SOFASCORE_ROLE_MAP.get(pos)
        if mapped and mapped not in roles:
            roles.append(mapped)

    return roles

def infer_primary_project_role(eligible_roles):
    if not eligible_roles:
        return pd.NA

    for role in ROLE_PRIORITY:
        if role in eligible_roles:
            return role

    return eligible_roles[0]


df = pd.read_csv(
    STATS_FILE,
    dtype={"season_year": str},
)
manifest = pd.read_csv(STAT_MANIFEST_FILE)
competition_manifest = pd.read_csv(
    COMPETITION_MANIFEST_FILE,
    dtype={"season_year": str},
)

weight_cols = [
    "competition_id",
    "season_id",
    "recency_weight",
    "competition_importance",
]

df = df.merge(
    competition_manifest[weight_cols],
    on=["competition_id", "season_id"],
    how="left"
)

df["recency_weight"] = df["recency_weight"].fillna(1.0)
df["competition_importance"] = df["competition_importance"].fillna(1.0)

df["aggregation_weight"] = (
    df["minutesPlayed"].fillna(0)
    * df["recency_weight"]
    * df["competition_importance"]
)

enabled = manifest[manifest["enabled"] == True]

identity_cols = enabled.loc[
    enabled["aggregation"] == "identity",
    "stat"
].tolist()

sum_cols = enabled.loc[
    enabled["aggregation"] == "sum",
    "stat"
].tolist()

weighted_cols = enabled.loc[
    enabled["aggregation"] == "weighted_mean",
    "stat"
].tolist()

for col in sum_cols + weighted_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "minutesPlayed" in df.columns:
    df["minutesPlayed"] = pd.to_numeric(df["minutesPlayed"], errors="coerce").fillna(0)

latest_identity = (
    df.sort_values(["player_id", "season_year"])
    .groupby("player_id", as_index=False)
    .tail(1)
)

identity_out = latest_identity[
    [col for col in identity_cols if col in latest_identity.columns]
].copy()

sum_data = (
    df.groupby("player_id", as_index=False)[
        [col for col in sum_cols if col in df.columns]
    ]
    .sum(min_count=1)
)

weighted_rows = []

for player_id, group in df.groupby("player_id"):
    row = {"player_id": player_id}

    weights = group["aggregation_weight"].fillna(0)

    for col in weighted_cols:
        if col not in group.columns:
            continue

        values = pd.to_numeric(group[col], errors="coerce")

        valid = values.notna() & weights.notna() & (weights > 0)

        if valid.any():
            row[col] = np.average(values[valid], weights=weights[valid])
        else:
            row[col] = values.mean()

    weighted_rows.append(row)

weighted_data = pd.DataFrame(weighted_rows)

out = identity_out.merge(sum_data, on="player_id", how="left")
out = out.merge(weighted_data, on="player_id", how="left")

print("=" * 60)
print("PROFILE MERGE AUDIT")
print(f"Rows after profile merge: {len(out):,}")

if "country" in out.columns:
    print(f"Missing country after merge: {out['country'].isna().sum():,}")
    print(f"Matched profiles: {out['country'].notna().sum():,}")

    missing_profiles = out[out["country"].isna()]

    cols = [
        c for c in [
            "player_id",
            "player",
            "source_competitions",
            "competition_count",
            "season_count",
        ]
        if c in missing_profiles.columns
    ]

    print("\nSample missing-profile rows:")
    print(missing_profiles[cols].head(20))

competition_counts = (
    df.groupby("player_id")["competition_id"]
      .nunique()
      .rename("competition_count")
)

season_counts = (
    df.groupby("player_id")["season_id"]
      .nunique()
      .rename("season_count")
)

out = out.merge(
    competition_counts,
    on="player_id",
    how="left",
)

out = out.merge(
    season_counts,
    on="player_id",
    how="left",
)

if PROFILES_FILE.exists() and PROFILES_FILE.stat().st_size > 0:
    profiles = pd.read_csv(PROFILES_FILE)

    profile_cols = [
        "player_id",
        "position",
        "positions_detailed",
        "height",
        "preferred_foot",
        "date_of_birth",
        "date_of_birth_timestamp",
        "country",
        "country_alpha2",
        "country_alpha3",
        "current_team_id",
        "current_team",
        "competition_count",
        "season_count",
    ]

    profiles = profiles[
        [col for col in profile_cols if col in profiles.columns]
    ].drop_duplicates(subset=["player_id"])

    out = out.merge(profiles, on="player_id", how="left")

    # Preserve original broad Sofascore position
    if "position" in out.columns:
        out = out.rename(columns={"position": "sofascore_position"})

    # Build project role fields
    if "positions_detailed" in out.columns:
        out["eligible_roles"] = out["positions_detailed"].apply(infer_eligible_roles)
        out["position"] = out["eligible_roles"].apply(infer_primary_project_role)

model_cols = manifest.loc[
    manifest["include_in_model"] == True,
    "stat"
].tolist()

profile_keep_cols = [
    "sofascore_position",
    "positions_detailed",
    "eligible_roles",
    "position",
    "height",
    "preferred_foot",
    "date_of_birth",
    "date_of_birth_timestamp",
    "country",
    "country_alpha2",
    "country_alpha3",
    "current_team_id",
    "current_team",
]

keep_cols = [
    col for col in out.columns
    if col in model_cols or col in profile_keep_cols
]

out = out[keep_cols]

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print(out.head())
print(out.shape)
print(f"Wrote: {OUT_FILE}")