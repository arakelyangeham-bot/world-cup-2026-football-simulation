from pathlib import Path
import ast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv"
ROLE_MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "role_feature_manifest.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"

ROLE_FEATURES = {
    "GK": ["rating", "saves_per90", "goalsPrevented", "cleanSheet_per90"],
    "CB": ["rating", "clearances_per90", "aerialDuelsWon_per90", "interceptions_per90"],
    "FB": ["rating", "tackles_per90", "interceptions_per90", "accurateCrosses_per90", "keyPasses_per90"],
    "DM": ["rating", "ballRecovery_per90", "tacklesWon_per90", "interceptions_per90", "totalPasses_per90"],
    "CM": ["rating", "totalPasses_per90", "keyPasses_per90", "accurateFinalThirdPasses_per90"],
    "AM": ["rating", "keyPasses_per90", "expectedAssists_per90", "successfulDribbles_per90"],
    "WM": ["rating", "accurateCrosses_per90", "keyPasses_per90", "ballRecovery_per90"],
    "W": ["rating", "expectedAssists_per90", "successfulDribbles_per90", "goals_per90"],
    "ST": ["rating", "expectedGoals_per90", "goals_per90", "shotsOnTarget_per90"],
}

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

df = pd.read_csv(IN_FILE)

if "eligible_roles" not in df.columns:
    raise KeyError(
        "eligible_roles column missing. Run scripts/build_role_mapping.py first."
    )

role_manifest = pd.read_csv(ROLE_MANIFEST_FILE)

role_manifest["weight"] = pd.to_numeric(
    role_manifest["weight"],
    errors="coerce"
).fillna(1.0)

df["eligible_roles_list"] = df["eligible_roles"].apply(parse_roles)

roles = sorted(role_manifest["role"].unique())

for role in roles:
    role_rows = role_manifest[role_manifest["role"] == role]

    score_parts = []
    weight_parts = []

    for _, feature_row in role_rows.iterrows():
        feature = feature_row["feature"]
        weight = feature_row["weight"]

        if feature not in df.columns:
            print(f"Skipping {role}: missing {feature}")
            continue

        z_col = f"z_{feature}"

        if z_col not in df.columns:
            df[z_col] = zscore(df[feature])

        score_parts.append(df[z_col] * weight)
        weight_parts.append(weight)

    if not score_parts:
        df[f"rating_{role}"] = np.nan
        continue

    raw_rating = sum(score_parts) / sum(weight_parts)

    is_eligible = df["eligible_roles_list"].apply(lambda eligible: role in eligible)

    df[f"raw_rating_{role}"] = np.where(
        is_eligible,
        raw_rating,
        np.nan,
    )

    df[f"rating_{role}"] = np.where(
        is_eligible,
        raw_rating * df["evidence_confidence"],
        np.nan,
    )

rating_cols = [f"rating_{role}" for role in roles]

df["best_rating"] = df[rating_cols].max(axis=1, skipna=True)

df["best_role"] = pd.NA

has_rating = df["best_rating"].notna()

df.loc[has_rating, "best_role"] = (
    df.loc[has_rating, rating_cols]
    .idxmax(axis=1)
    .str.replace("rating_", "", regex=False)
)

out_cols = [
    "player_id",
    "player",
    "team",
    "current_team",
    "country",
    "position",
    "positions_detailed",
    "eligible_roles",
    "minutesPlayed",
    "rating",
] + rating_cols + [
    "best_role",
    "best_rating",
]

out_cols = [c for c in out_cols if c in df.columns]

out = df[out_cols].copy()

out.to_csv(OUT_FILE, index=False)

print(out.head())
print(out.shape)
print(f"Wrote: {OUT_FILE}")