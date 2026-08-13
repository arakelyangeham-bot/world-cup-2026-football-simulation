#audit_role_rating_coverage.py

from pathlib import Path
import ast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "role_rating_coverage.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

ROLES = ["GK", "CB", "FB", "DM", "CM", "AM", "WM", "W", "ST"]


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


df = pd.read_csv(RATINGS_FILE)
df["eligible_roles_list"] = df["eligible_roles"].apply(parse_roles)

rows = []

for country, country_df in df.groupby("country"):
    row = {
        "country": country,
        "players": len(country_df),
    }

    for role in ROLES:
        rating_col = f"rating_{role}"

        eligible = country_df["eligible_roles_list"].apply(
            lambda roles: role in roles
        )

        row[f"eligible_{role}"] = eligible.sum()
        row[f"rated_{role}"] = (
            country_df.loc[eligible, rating_col].notna().sum()
            if rating_col in country_df.columns
            else 0
        )

    rows.append(row)

out = pd.DataFrame(rows)

out["total_eligible"] = out[[f"eligible_{r}" for r in ROLES]].sum(axis=1)
out["total_rated"] = out[[f"rated_{r}" for r in ROLES]].sum(axis=1)
out["rating_gap"] = out["total_eligible"] - out["total_rated"]

out = out.sort_values(
    ["rating_gap", "players"],
    ascending=[False, True],
)

out.to_csv(OUT_FILE, index=False)

print(out.head(40))
print(f"Wrote: {OUT_FILE}")