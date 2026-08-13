#audit_lineup_coverage.py

from pathlib import Path
import ast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"
LINEUPS_FILE = PROJECT_ROOT / "data" / "processed" / "expected_lineups.csv"

OUT_FILE = PROJECT_ROOT / "outputs" / "lineup_coverage.csv"
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


ratings = pd.read_csv(RATINGS_FILE)
lineups = pd.read_csv(LINEUPS_FILE)

ratings["eligible_roles_list"] = ratings["eligible_roles"].apply(parse_roles)

rows = []

for country, country_df in ratings.groupby("country"):
    lineup_df = lineups[lineups["country"] == country]

    row = {
        "country": country,
        "rated_players": len(country_df),
        "selected_players": lineup_df["player_id"].notna().sum(),
        "missing_slots": lineup_df["player_id"].isna().sum(),
    }

    for role in ROLES:
        row[f"eligible_{role}"] = country_df["eligible_roles_list"].apply(
            lambda roles: role in roles
        ).sum()

    rows.append(row)

out = pd.DataFrame(rows).sort_values(
    ["missing_slots", "rated_players"],
    ascending=[False, True],
)

out.to_csv(OUT_FILE, index=False)

print(out.head(40))
print(f"Wrote: {OUT_FILE}")