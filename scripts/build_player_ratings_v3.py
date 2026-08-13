#build_player_ratings_v3.py

from pathlib import Path
import ast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_stats.csv"
PROFILES_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_profiles.csv"
COMPETITION_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "competition_manifest.csv"
COMPETITION_FEATURE_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "competition_feature_manifest.csv"
REGISTRY_FILE = PROJECT_ROOT / "data" / "processed" / "player_registry.csv"
ROLE_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "role_feature_manifest.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"


def parse_roles(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def zscore(series):
    series = pd.to_numeric(series, errors="coerce")
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


stats = pd.read_csv(STATS_FILE, dtype={"season_year": str})
profiles = pd.read_csv(PROFILES_FILE)
competitions = pd.read_csv(COMPETITION_MANIFEST_FILE, dtype={"season_year": str})
registry = pd.read_csv(REGISTRY_FILE)
role_manifest = pd.read_csv(ROLE_MANIFEST_FILE)
competition_features = pd.read_csv(
    COMPETITION_FEATURE_FILE,
    dtype={"season_year": str},
)

stats = stats.merge(
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

stats = stats.merge(
    profiles[
        [
            "player_id",
            "country",
            "position",
            "positions_detailed",
            "current_team",
        ]
    ],
    on="player_id",
    how="left",
)

stats = stats.merge(
    registry[["player_id", "canonical_player_id", "eligible_roles"]],
    on="player_id",
    how="left",
)

stats["canonical_player_id"] = stats["canonical_player_id"].fillna(stats["player_id"])
stats["eligible_roles"] = stats["eligible_roles"].fillna("[]")
stats["eligible_roles_list"] = stats["eligible_roles"].apply(parse_roles)

stats["row_weight"] = (
    pd.to_numeric(stats["minutesPlayed"], errors="coerce").fillna(0)
    * pd.to_numeric(stats["recency_weight"], errors="coerce").fillna(1)
    * pd.to_numeric(stats["competition_importance"], errors="coerce").fillna(1)
)

role_manifest["weight"] = pd.to_numeric(
    role_manifest["weight"],
    errors="coerce",
).fillna(1.0)

roles = sorted(role_manifest["role"].unique())

for feature in role_manifest["feature"].unique():
    if feature in stats.columns:
        stats[f"z_{feature}"] = zscore(stats[feature])

rating_rows = []

for canonical_player_id, group in stats.groupby("canonical_player_id"):
    base = group.iloc[-1]

    row = {
        "canonical_player_id": canonical_player_id,
        "player_id": base["player_id"],
        "player": base.get("player"),
        "country": group["country"].dropna().iloc[-1] if group["country"].notna().any() else pd.NA,
        "current_team": group["current_team"].dropna().iloc[-1] if group["current_team"].notna().any() else pd.NA,
        "position": group["position"].dropna().iloc[-1] if group["position"].notna().any() else pd.NA,
        "positions_detailed": group["positions_detailed"].dropna().iloc[-1] if group["positions_detailed"].notna().any() else pd.NA,
        "eligible_roles": group["eligible_roles"].dropna().iloc[-1],
        "minutesPlayed": pd.to_numeric(group["minutesPlayed"], errors="coerce").sum(),
        "total_weighted_evidence": group["row_weight"].sum(),
        "source_competitions": "; ".join(sorted(set(group["competition"].dropna()))),
        "competition_count": group["competition_id"].nunique(),
        "season_count": group["season_id"].nunique(),
    }

    eligible_roles = parse_roles(row["eligible_roles"])
    evidence_confidence = min(row["total_weighted_evidence"] / 1800, 1.0)
    row["evidence_confidence"] = evidence_confidence

    for role in roles:
        if role not in eligible_roles:
            row[f"raw_rating_{role}"] = np.nan
            row[f"rating_{role}"] = np.nan
            continue

        role_rows = role_manifest[role_manifest["role"] == role]

        score_parts = []
        used_weight_parts = []

        for _, feature_row in role_rows.iterrows():
            feature = feature_row["feature"]
            feature_weight = feature_row["weight"]
            z_col = f"z_{feature}"

            if z_col not in group.columns:
                continue

            availability = competition_features[
                (competition_features["feature"] == feature)
            ][["competition", "season_year", "available"]]

            feature_group = group.merge(
                availability,
                on=["competition", "season_year"],
                how="left",
            )

            feature_group["available"] = feature_group["available"].fillna(True)

            values = feature_group[z_col]
            weights = feature_group["row_weight"]

            valid = (
                values.notna()
                & weights.notna()
                & (weights > 0)
                & feature_group["available"]
            )

            if not valid.any():
                continue

            feature_score = np.average(values[valid], weights=weights[valid])

            score_parts.append(feature_score * feature_weight)
            used_weight_parts.append(feature_weight)

        if not score_parts:
            row[f"raw_rating_{role}"] = np.nan
            row[f"rating_{role}"] = np.nan
            continue

        raw_rating = sum(score_parts) / sum(used_weight_parts)

        row[f"raw_rating_{role}"] = raw_rating
        row[f"rating_{role}"] = raw_rating * evidence_confidence

    rating_rows.append(row)

out = pd.DataFrame(rating_rows)

rating_cols = [f"rating_{role}" for role in roles]

out["best_rating"] = out[rating_cols].max(axis=1, skipna=True)
out["best_role"] = pd.NA

has_rating = out["best_rating"].notna()

out.loc[has_rating, "best_role"] = (
    out.loc[has_rating, rating_cols]
    .idxmax(axis=1)
    .str.replace("rating_", "", regex=False)
)

out.to_csv(OUT_FILE, index=False)

print(out.head())
print(out.shape)
print(f"Wrote: {OUT_FILE}")