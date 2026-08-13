# feature_sets.py

"""
Central definitions for all model feature sets.

These feature sets are reused by every model
(Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Net, etc.)
to ensure fair comparisons.
"""

# ---------------------------------------------------------------------
# Base Team Ratings
# ---------------------------------------------------------------------

RAW_FEATURES = [
    "home_attack",
    "home_midfield",
    "home_defense",
    "home_gk",
    "away_attack",
    "away_midfield",
    "away_defense",
    "away_gk",
]

# ---------------------------------------------------------------------
# Engineered Rating Differences
# ---------------------------------------------------------------------

ENGINEERED_FEATURES = [
    "attack_diff",
    "midfield_diff",
    "defense_diff",
    "gk_diff",
]

# ---------------------------------------------------------------------
# Match Engine Features
# ---------------------------------------------------------------------

ENGINE_FEATURES = [
    "home_poisson_attack",
    "home_poisson_defense",
    "away_poisson_attack",
    "away_poisson_defense",
    "poisson_attack_diff",
    "poisson_defense_diff",
]

# ---------------------------------------------------------------------
# National-Team Priors
# ---------------------------------------------------------------------

PRIOR_FEATURES = [
    "fifa_points_diff",
]

# ---------------------------------------------------------------------
# Combined Feature Set
# ---------------------------------------------------------------------

CORE_FEATURES = (
    RAW_FEATURES
    + ENGINEERED_FEATURES
    + ENGINE_FEATURES
)

FEATURES_V1 = CORE_FEATURES

FEATURES_V2 = (
    CORE_FEATURES
    + PRIOR_FEATURES
)

FEATURE_SETS = {
    # Frozen experiment snapshots
    "v1": FEATURES_V1,
    "v2": FEATURES_V2,
}


def get_feature_set(name: str):
    """
    Retrieve a named feature set.
    """

    if name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{name}'. "
            f"Available: {list(FEATURE_SETS.keys())}"
        )

    return FEATURE_SETS[name]