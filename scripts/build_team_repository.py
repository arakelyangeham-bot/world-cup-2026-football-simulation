#build_team_repository.py

from pathlib import Path
import pandas as pd
from shared.national_team_priors import load_fifa_points
from shared.team_name_normalizer import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINEUPS_FILE = PROJECT_ROOT / "data" / "processed" / "expected_lineups.csv"
RATINGS_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"


ATTACK_ROLES = {"ST", "W", "AM"}
MIDFIELD_ROLES = {"DM", "CM", "WM"}
DEFENSE_ROLES = {"CB", "FB"}
GK_ROLES = {"GK"}


def safe_mean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if values.empty:
        return 0.0

    return float(values.mean())


lineups = pd.read_csv(LINEUPS_FILE)
ratings = pd.read_csv(RATINGS_FILE)

rows = []

for country, group in lineups.groupby("country"):
    attack_scores = []
    midfield_scores = []
    defense_scores = []
    gk_scores = []

    for _, row in group.iterrows():
        role = row["role"]
        rating = row["rating"]

        if role in ATTACK_ROLES:
            attack_scores.append(rating)
        elif role in MIDFIELD_ROLES:
            midfield_scores.append(rating)
        elif role in DEFENSE_ROLES:
            defense_scores.append(rating)
        elif role in GK_ROLES:
            gk_scores.append(rating)

    att = safe_mean(attack_scores)
    mid = safe_mean(midfield_scores)
    defense = safe_mean(defense_scores)
    gk = safe_mean(gk_scores)

    # First simple bridge into the existing Poisson schema.
    poisson_attack = 1.0 + (att * 0.10) + (mid * 0.04)
    poisson_defense = 1.0 - (defense * 0.08) - (gk * 0.04)

    poisson_attack = max(0.50, min(1.50, poisson_attack))
    poisson_defense = max(0.50, min(1.50, poisson_defense))

    rows.append({
        "nation": country,
        "att_composite": att,
        "mid_composite": mid,
        "def_composite": defense,
        "gk_composite": gk,
        "poisson_attack_adj": poisson_attack,
        "poisson_defense_adj": poisson_defense,
    })

out = pd.DataFrame(rows)

fifa_points = load_fifa_points()
valid_nations = set(fifa_points)

out["normalized_nation"] = out["nation"].map(normalize_team_name)

out = out[out["normalized_nation"].isin(valid_nations)].copy()

out = out.drop(columns=["normalized_nation"])

out.to_csv(OUT_FILE, index=False)

print(out.head(30))
print(out.shape)
print(f"Wrote: {OUT_FILE}")