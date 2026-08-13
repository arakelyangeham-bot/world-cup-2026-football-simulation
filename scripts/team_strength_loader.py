#scripts/team_strength_loader.py

from pathlib import Path
import pandas as pd
from shared.national_team_priors import load_fifa_points
from shared.team_name_normalizer import normalize_team_name
from simulation.simulation_config import TEAM_REPOSITORY_SOURCE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAM_STRENGTH_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"
FIFA_RANKINGS_FILE = PROJECT_ROOT / "data" / "external" / "fifa_rankings.csv"

REPOSITORY_PATHS = {
    "legacy": TEAM_STRENGTH_FILE,
    "dimension_specific": (
        PROJECT_ROOT
        / "outputs"
        / "study_011_team_representation_calibration"
        / "repositories"
        / "dimension_specific_team_repository.csv"
    ),
    "top_11_mean": (
        PROJECT_ROOT
        / "outputs"
        / "study_011_team_representation_calibration"
        / "repositories"
        / "top_11_mean_team_repository.csv"
    ),
    "top_5_mean": (
        PROJECT_ROOT
        / "outputs"
        / "study_011_team_representation_calibration"
        / "repositories"
        / "top_5_mean_team_repository.csv"
    ),
    "star_weighted": (
        PROJECT_ROOT
        / "outputs"
        / "study_011_team_representation_calibration"
        / "repositories"
        / "star_weighted_team_repository.csv"
    ),
    "starter_plus_depth":(
        PROJECT_ROOT
        / "outputs"
        / "study_011_team_representation_calibration"
        / "repositories"
        / "starter_plus_depth_team_repository.csv"
    ),
    "premier_league_validation": (
        PROJECT_ROOT
        / "data"
        / "team_repositories"
        / "premier_league_validation_repository.csv"
    ),
    "premier_league_production_v1": (
        PROJECT_ROOT
        / "outputs"
        / "study_071a_premier_league_club_repository_v1"
        / "premier_league_club_repository_v1.csv"
    ),
}

def load_team_ratings() -> dict[str, float]:
    df = pd.read_csv(TEAM_STRENGTH_FILE)

    required_cols = {
        "nation",
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required team strength columns: {missing}")

    df["raw_strength"] = (
        df["att_composite"].fillna(0) * 0.35
        + df["mid_composite"].fillna(0) * 0.25
        + df["def_composite"].fillna(0) * 0.25
        + df["gk_composite"].fillna(0) * 0.15
    )

    df["rating"] = 1500 + df["raw_strength"] * 100

    return dict(zip(df["nation"], df["rating"]))

def load_poisson_team_strengths() -> dict[str, dict[str, float]]:
    df = pd.read_csv(TEAM_STRENGTH_FILE)

    required_cols = {
        "nation",
        "poisson_attack_adj",
        "poisson_defense_adj",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required Poisson columns: {missing}")

    strengths = {}

    for _, row in df.iterrows():
        nation = row["nation"]

        attack = row["poisson_attack_adj"]
        defense = row["poisson_defense_adj"]

        strengths[nation] = {
            "attack": 1.0 if pd.isna(attack) else float(attack),
            "defense": 1.0 if pd.isna(defense) else float(defense),
        }

    return strengths

def load_complete_team_strengths() -> dict[str, dict]:
    """
    Load all team-level priors into one unified dictionary.

    This is intended to become the canonical feature store for
    national-team attributes.
    """

    strengths = pd.read_csv(TEAM_STRENGTH_FILE)
    fifa_points = load_fifa_points()

    complete = {}

    for _, row in strengths.iterrows():
        nation = normalize_team_name(row["nation"])

        complete[nation] = {
            "att_composite": float(row["att_composite"]),
            "mid_composite": float(row["mid_composite"]),
            "def_composite": float(row["def_composite"]),
            "gk_composite": float(row["gk_composite"]),

            "poisson_attack_adj": float(row["poisson_attack_adj"]),
            "poisson_defense_adj": float(row["poisson_defense_adj"]),

            "fifa_points": fifa_points.get(nation),
        }

    return complete

def load_team_repository(path: Path | None = None) -> dict[str, dict]:
    """
    Load the canonical team repository.

    This is the single internal representation that should be used by
    both Poisson and ML-guided simulation code.

    Public canonical keys:
        attack
        midfield
        defense
        gk
        poisson_attack
        poisson_defense
        rating_prior

    Temporary compatibility aliases:
        fifa_points

    Legacy aliases are also included temporarily for backward compatibility.
    """

    if path is None:
        if TEAM_REPOSITORY_SOURCE not in REPOSITORY_PATHS:
            raise ValueError(
                f"Unknown TEAM_REPOSITORY_SOURCE: {TEAM_REPOSITORY_SOURCE}"
            )

        repository_path = REPOSITORY_PATHS[TEAM_REPOSITORY_SOURCE]
    else:
        repository_path = path

    strengths = pd.read_csv(repository_path)
    
    fifa_points = load_fifa_points()

    repository = {}

    for _, row in strengths.iterrows():
        nation = normalize_team_name(row["nation"])

        nation_fifa_points = fifa_points.get(nation)

        if nation_fifa_points is None:
            raise ValueError(
            f"Missing FIFA points for team in repository: {nation}"
        )

        attack = float(row["att_composite"])
        midfield = float(row["mid_composite"])
        defense = float(row["def_composite"])
        gk = float(row["gk_composite"])

        poisson_attack = float(row["poisson_attack_adj"])
        poisson_defense = float(row["poisson_defense_adj"])

        repository[nation] = {
            # Canonical public schema
            "attack": attack,
            "midfield": midfield,
            "defense": defense,
            "gk": gk,
            "poisson_attack": poisson_attack,
            "poisson_defense": poisson_defense,

            # Generic external team-strength prior
            "rating_prior": nation_fifa_points,

            # Temporary national-team compatibility alias
            "fifa_points": nation_fifa_points,

            # Temporary legacy aliases
            "att_composite": attack,
            "mid_composite": midfield,
            "def_composite": defense,
            "gk_composite": gk,
            "poisson_attack_adj": poisson_attack,
            "poisson_defense_adj": poisson_defense,
        }

    return repository

if __name__ == "__main__":
    ratings = load_team_ratings()

    print(f"Loaded ratings for {len(ratings)} teams")

    for team, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{team}: {rating:.1f}")

        print()
        
    poisson_strengths = load_poisson_team_strengths()
    print(f"Loaded Poisson strengths for {len(poisson_strengths)} teams")

    for team, vals in list(poisson_strengths.items())[:10]:
        print(team, vals)
    
    fifa_points = load_fifa_points()
    print()
    print(f"Loaded FIFA points for {len(fifa_points)} teams")

    for team, points in list(fifa_points.items())[:10]:
        print(team, points)
    
    
    complete = load_complete_team_strengths()

    print()
    print(f"Loaded complete strengths for {len(complete)} teams")

    france = complete["France"]

    print()
    print("France")
    for key, value in france.items():
        print(f"  {key}: {value}")

    repository = load_team_repository()

    print()
    print(f"Loaded canonical team repository for {len(repository)} teams")

    france = repository["France"]

    print()
    print("France canonical schema")
    for key, value in france.items():
        print(f"  {key}: {value}")