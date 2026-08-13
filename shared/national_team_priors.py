#national_team_priors.py

import pandas as pd

from shared.config import PROJECT_ROOT
from shared.team_name_normalizer import normalize_team_name

FIFA_RANKINGS_PATH = PROJECT_ROOT / "data" / "external" / "fifa_rankings.csv"


def load_fifa_points(path=FIFA_RANKINGS_PATH):
    df = pd.read_csv(path)

    required = {"team", "points"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing FIFA ranking columns: {missing}")

    df["team"] = df["team"].map(normalize_team_name)
    df["fifa_points"] = pd.to_numeric(df["points"], errors="raise") 

    return {
        row["team"]: float(row["fifa_points"])
        for _, row in df.iterrows()
    }


def load_national_team_priors():
    fifa_points = load_fifa_points()

    return {
        team: {"fifa_points": points}
        for team, points in fifa_points.items()
    }