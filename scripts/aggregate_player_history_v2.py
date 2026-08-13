# ARCHITECTURE NOTE:
#
# This script is retained as a general/legacy historical aggregation utility.
#
# It is NOT part of the modern Player Intelligence production path.
# Player Intelligence must preserve competition-season evidence grain until
# historical competition/recency/availability weighting has occurred.
#
# Current Player Intelligence weighting stage:
#
#     scripts.build_weighted_player_features
#
# Do not insert this historical aggregation stage before
# scripts.sofascore_feature_engineering in the modern Player Intelligence
# pipeline.

from pathlib import Path
import ast
import numpy as np
import pandas as pd

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"
STAT_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "stat_manifest.csv"
COMPETITION_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "competition_manifest.csv"
PROFILES_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_profiles.csv"
PLAYER_ID_ALIASES_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_id_aliases.csv"
)
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_player_dataset.csv"

ROLE_PRIORITY = ["GK", "ST", "W", "AM", "CM", "DM", "FB", "CB", "WM"]

ROLE_MAP = {
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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate competition-season player statistics "
            "into one historical evidence row per player."
        )
    )

    parser.add_argument(
        "--stats-file",
        type=Path,
        default=STATS_FILE,
    )

    parser.add_argument(
        "--competition-manifest-file",
        type=Path,
        default=COMPETITION_MANIFEST_FILE,
    )

    parser.add_argument(
        "--stat-manifest-file",
        type=Path,
        default=STAT_MANIFEST_FILE,
    )

    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=PROFILES_FILE,
    )

    parser.add_argument(
        "--player-id-aliases-file",
        type=Path,
        default=PLAYER_ID_ALIASES_FILE,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUT_FILE,
    )

    return parser.parse_args()

def infer_eligible_roles(positions_detailed):
    roles = []

    for pos in parse_positions(positions_detailed):
        role = ROLE_MAP.get(pos)
        if role and role not in roles:
            roles.append(role)

    return roles


def infer_primary_role(eligible_roles):
    if not eligible_roles:
        return pd.NA

    for role in ROLE_PRIORITY:
        if role in eligible_roles:
            return role

    return eligible_roles[0]


def weighted_mean(group, value_col, weight_col):
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0)

    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.any():
        return np.average(values[valid], weights=weights[valid])

    return values.mean()

arguments = parse_arguments()

stats_file = arguments.stats_file
competition_manifest_file = (
    arguments.competition_manifest_file
)
stat_manifest_file = (
    arguments.stat_manifest_file
)
profiles_file = arguments.profiles_file
player_id_aliases_file = (
    arguments.player_id_aliases_file
)
output_file = arguments.output_file

stats = pd.read_csv(
    stats_file,
    dtype={"season_year": str},
)

stat_manifest = pd.read_csv(
    stat_manifest_file
)

competition_manifest = pd.read_csv(
    competition_manifest_file,
    dtype={"season_year": str},
)

stats = stats.merge(
    competition_manifest[
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

stats["recency_weight"] = stats["recency_weight"].fillna(1.0)
stats["competition_importance"] = stats["competition_importance"].fillna(1.0)
stats["minutesPlayed"] = pd.to_numeric(
    stats["minutesPlayed"],
    errors="coerce",
).fillna(0)

stats["aggregation_weight"] = (
    stats["minutesPlayed"]
    * stats["recency_weight"]
    * stats["competition_importance"]
)

enabled = stat_manifest[stat_manifest["enabled"] == True]

sum_cols = enabled.loc[
    enabled["aggregation"] == "sum",
    "stat",
].tolist()

weighted_cols = enabled.loc[
    enabled["aggregation"] == "weighted_mean",
    "stat",
].tolist()

sum_cols = [
    col for col in sum_cols
    if col in stats.columns and col != "player_id"
]

weighted_cols = [
    col for col in weighted_cols
    if col in stats.columns and col != "player_id"
]

for col in sum_cols + weighted_cols:
    stats[col] = pd.to_numeric(stats[col], errors="coerce")

# 1. Sum-based stats
sum_data = (
    stats.groupby("player_id", as_index=False)[sum_cols]
    .sum(min_count=1)
)

# 2. Weighted-mean stats
weighted_rows = []

for player_id, group in stats.groupby("player_id"):
    row = {"player_id": player_id}

    for col in weighted_cols:
        row[col] = weighted_mean(group, col, "aggregation_weight")

    weighted_rows.append(row)

weighted_data = pd.DataFrame(weighted_rows)

# 3. Source coverage metadata
source_meta = (
    stats.groupby("player_id")
    .agg(
        source_competitions=(
            "competition",
            lambda x: "; ".join(sorted(set(x.dropna()))),
        ),
        competition_count=("competition_id", "nunique"),
        season_count=("season_id", "nunique"),
        total_evidence_minutes=("minutesPlayed", "sum"),
        total_weighted_evidence=("aggregation_weight", "sum"),
    )
    .reset_index()
)



# 4. Start canonical one-row-per-player table
out = sum_data.merge(
    weighted_data,
    on="player_id",
    how="outer",
)

out = out.merge(
    source_meta,
    on="player_id",
    how="left",
)

# Resolve profile lookup IDs without rewriting raw source identity.
out["profile_player_id"] = (
    pd.to_numeric(
        out["player_id"],
        errors="raise",
    )
    .astype(int)
)

if (
    player_id_aliases_file.exists()
    and player_id_aliases_file.stat().st_size > 0
):
    aliases = pd.read_csv(
        player_id_aliases_file
    )

    required_alias_columns = {
        "source_player_id",
        "canonical_player_id",
        "review_status",
    }

    missing_alias_columns = (
        required_alias_columns
        - set(aliases.columns)
    )

    if missing_alias_columns:
        raise ValueError(
            "Player ID alias file is missing required columns: "
            f"{sorted(missing_alias_columns)}"
        )

    reviewed_aliases = aliases[
        aliases["review_status"] == "reviewed"
    ].copy()

    if reviewed_aliases["source_player_id"].duplicated().any():
        duplicate_ids = (
            reviewed_aliases.loc[
                reviewed_aliases[
                    "source_player_id"
                ].duplicated(keep=False),
                "source_player_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate reviewed source player IDs in alias file: "
            f"{duplicate_ids}"
        )

    alias_map = dict(
        zip(
            reviewed_aliases["source_player_id"].astype(int),
            reviewed_aliases["canonical_player_id"].astype(int),
        )
    )

    out["profile_player_id"] = (
        out["profile_player_id"]
        .map(alias_map)
        .fillna(out["profile_player_id"])
        .astype(int)
    )

# 5. Merge player profiles
profiles = pd.read_csv(
    profiles_file
)

profile_cols = [
    "player_id",
    "player",
    "player_slug",
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
]

profiles = profiles[
    [col for col in profile_cols if col in profiles.columns]
].drop_duplicates(subset=["player_id"])

profiles = profiles.rename(
    columns={
        "player_id": "profile_player_id",
        "position": "sofascore_position",
    }
)

out = out.merge(
    profiles,
    on="profile_player_id",
    how="left",
)

# 6. Project role fields
if "positions_detailed" in out.columns:
    out["eligible_roles"] = out["positions_detailed"].apply(infer_eligible_roles)
    out["position"] = out["eligible_roles"].apply(infer_primary_role)

# 7. Final cleanup
output_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    output_file,
    index=False,
)

print(out.head())
print(out.shape)
print(f"Unique players: {out['player_id'].nunique()}")
print(f"Missing country: {out['country'].isna().sum() if 'country' in out.columns else 'N/A'}")
print(f"Wrote: {output_file}")