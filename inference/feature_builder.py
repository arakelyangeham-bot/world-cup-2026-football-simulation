# inference/feature_builder.py

import pandas as pd

from shared.feature_sets import get_feature_set
from inference.expected_goals_features import (
    ExpectedGoalsFeatures,
)

PRODUCTION_FEATURE_SET = "v2"
PRODUCTION_FEATURES = get_feature_set(PRODUCTION_FEATURE_SET)


def build_engineered_features(home_team, away_team):
    """
    Build the full production feature row for one match.

    This must stay aligned with shared.feature_sets.FEATURES_V2.
    """

    feature_row = {
        # Raw team ratings
        "home_attack": home_team["attack"],
        "home_midfield": home_team["midfield"],
        "home_defense": home_team["defense"],
        "home_gk": home_team["gk"],

        "away_attack": away_team["attack"],
        "away_midfield": away_team["midfield"],
        "away_defense": away_team["defense"],
        "away_gk": away_team["gk"],

        # Engineered differences
        "attack_diff": home_team["attack"] - away_team["attack"],
        "midfield_diff": home_team["midfield"] - away_team["midfield"],
        "defense_diff": home_team["defense"] - away_team["defense"],
        "gk_diff": home_team["gk"] - away_team["gk"],

        # Poisson engine features
        "home_poisson_attack": home_team["poisson_attack"],
        "home_poisson_defense": home_team["poisson_defense"],
        "away_poisson_attack": away_team["poisson_attack"],
        "away_poisson_defense": away_team["poisson_defense"],

        "poisson_attack_diff": (
            home_team["poisson_attack"]
            - away_team["poisson_attack"]
        ),
        "poisson_defense_diff": (
            home_team["poisson_defense"]
            - away_team["poisson_defense"]
        ),

        # National-team prior
        "fifa_points_diff": (
            home_team["fifa_points"]
            - away_team["fifa_points"]
        ),
    }

    validate_feature_row(feature_row)

    return ExpectedGoalsFeatures(
        **{
            feature: feature_row[feature]
            for feature in PRODUCTION_FEATURES
        }
)


def validate_feature_row(feature_row):
    missing = [
        feature
        for feature in PRODUCTION_FEATURES
        if feature not in feature_row
    ]

    if missing:
        raise ValueError(f"Missing production features: {missing}")

    null_features = [
        feature
        for feature in PRODUCTION_FEATURES
        if pd.isna(feature_row[feature])
    ]

    if null_features:
        raise ValueError(f"Null production features: {null_features}")

    extra = [
        feature
        for feature in feature_row
        if feature not in PRODUCTION_FEATURES
    ]

    if extra:
        raise ValueError(f"Unexpected production features: {extra}")

    return True