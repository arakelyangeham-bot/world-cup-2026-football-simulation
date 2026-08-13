from pathlib import Path
import ast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_player_dataset.csv"
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


df = pd.read_csv(IN_FILE, dtype={"season_year": str})
role_manifest = pd.read_csv(ROLE_MANIFEST_FILE)

registry = pd.read_csv(REGISTRY_FILE)

df = df.merge(
    registry[["player_id", "canonical_player_id"]],
    on="player_id",
    how="left",
)

df["canonical_player_id"] = df["canonical_player_id"].fillna(df["player_id"])

df["eligible_roles_list"] = df["eligible_roles"].apply(parse_roles)

if "aggregation_weight" in df.columns:
    df["row_weight"] = pd.to_numeric(df["aggregation_weight"], errors="coerce").fillna(0)
else:
    df["row_weight"] = pd.to_numeric(df["minutesPlayed"], errors="coerce").fillna(0)
role_manifest["weight"] = pd.to_numeric(
    role_manifest["weight"],
    errors="coerce"
).fillna(1.0)

roles = sorted(role_manifest["role"].unique())

for feature in role_manifest["feature"].unique():
    if feature in df.columns:
        df[f"z_{feature}"] = zscore(df[feature])

rating_rows = []

for canonical_player_id, group in df.groupby("canonical_player_id"):
    base = group.iloc[-1].to_dict()

    row = {
        "canonical_player_id": canonical_player_id,
        "player_id": group["player_id"].iloc[-1],
        "player": group["player"].dropna().iloc[-1] if "player" in group else pd.NA,
        "country": group["country"].dropna().iloc[-1] if "country" in group and group["country"].notna().any() else pd.NA,
        "current_team": group["current_team"].dropna().iloc[-1] if "current_team" in group and group["current_team"].notna().any() else pd.NA,
        "position": group["position"].dropna().iloc[-1] if "position" in group and group["position"].notna().any() else pd.NA,
        "positions_detailed": group["positions_detailed"].dropna().iloc[-1] if "positions_detailed" in group and group["positions_detailed"].notna().any() else pd.NA,
        "eligible_roles": group["eligible_roles"].dropna().iloc[-1] if "eligible_roles" in group and group["eligible_roles"].notna().any() else "[]",
        "minutesPlayed": group["minutesPlayed"].sum(),
        "total_weighted_evidence": group["row_weight"].sum(),
        "source_competitions": "; ".join(sorted(set(group["competition"].dropna()))) if "competition" in group else pd.NA,
        "competition_count": group["competition_id"].nunique() if "competition_id" in group else pd.NA,
        "season_count": group["season_id"].nunique() if "season_id" in group else pd.NA,
    }

    eligible_roles = parse_roles(row["eligible_roles"])

    evidence_confidence = min(row["total_weighted_evidence"] / 1800, 1.0)
    row["evidence_confidence"] = evidence_confidence

    for role in roles:
        role_rows = role_manifest[role_manifest["role"] == role]

        if role not in eligible_roles:
            row[f"raw_rating_{role}"] = np.nan
            row[f"rating_{role}"] = np.nan
            continue

        score_parts = []
        weight_parts = []

        for _, feature_row in role_rows.iterrows():
            feature = feature_row["feature"]
            feature_weight = feature_row["weight"]
            z_col = f"z_{feature}"

            if z_col not in group.columns:
                continue

            values = group[z_col]
            weights = group["row_weight"]

            valid = values.notna() & weights.notna() & (weights > 0)

            if not valid.any():
                continue

            feature_score = np.average(values[valid], weights=weights[valid])
            score_parts.append(feature_score * feature_weight)
            weight_parts.append(feature_weight)

        if not score_parts:
            row[f"raw_rating_{role}"] = np.nan
            row[f"rating_{role}"] = np.nan
            continue

        raw_rating = sum(score_parts) / sum(weight_parts)

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