"""
Central feature-set definitions for model training experiments.

This keeps Version 1, Version 2, and future feature sets reproducible.
"""

BASE_FEATURES_V1 = [
    "attack_diff",
    "midfield_diff",
    "defense_diff",
    "gk_diff",
    "poisson_attack_diff",
    "poisson_defense_diff",
]

FEATURES_V2_FIFA_POINTS = BASE_FEATURES_V1 + [
    "fifa_points_diff",
]


FEATURE_SETS = {
    "v1_base": BASE_FEATURES_V1,
    "v2_fifa_points": FEATURES_V2_FIFA_POINTS,
}


DEFAULT_FEATURE_SET = "v2_fifa_points"


def get_feature_columns(feature_set_name: str = DEFAULT_FEATURE_SET) -> list[str]:
    if feature_set_name not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(
            f"Unknown feature set: {feature_set_name}. "
            f"Valid feature sets: {valid}"
        )

    return FEATURE_SETS[feature_set_name]