# fifa_rankings_loader.py

import pandas as pd

from shared.config import PROJECT_ROOT


FIFA_RANKINGS_PATH = PROJECT_ROOT / "data" / "external" / "fifa_rankings.csv"


NAME_ALIASES = {
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def normalize_team_name(name):
    name = str(name).strip()
    return NAME_ALIASES.get(name, name)


def load_fifa_rankings(path=FIFA_RANKINGS_PATH):
    df = pd.read_csv(path)

    required = {"team", "rank", "points"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing FIFA ranking columns: {missing}")

    df["team"] = df["team"].map(normalize_team_name)
    df["rank"] = pd.to_numeric(df["rank"], errors="raise")
    df["points"] = pd.to_numeric(df["points"], errors="raise")

    return {
        row["team"]: {
            "fifa_rank": int(row["rank"]),
            "fifa_points": float(row["points"]),
        }
        for _, row in df.iterrows()
    }