from pathlib import Path
import ast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv"

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
        role = ROLE_MAP.get(pos)
        if role and role not in roles:
            roles.append(role)

    return roles


df = pd.read_csv(IN_FILE)

df["eligible_roles"] = df["positions_detailed"].apply(infer_eligible_roles)

df.to_csv(OUT_FILE, index=False)

print(df[["player", "position", "positions_detailed", "eligible_roles"]].head(20))
print(f"Wrote: {OUT_FILE}")