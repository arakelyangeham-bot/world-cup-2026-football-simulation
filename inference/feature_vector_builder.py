#feature_vector_builder.py

import numpy as np

from inference.feature_builder import build_engineered_features


def get_feature_order(model) -> list[str]:
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not expose feature_names_in_. "
            "Cannot build safe NumPy feature vectors."
        )

    return list(model.feature_names_in_)


def build_feature_vector(
    home_team: dict,
    away_team: dict,
    feature_order: list[str],
) -> np.ndarray:
    feature_row = build_engineered_features(home_team, away_team)

    return np.array(
        [[feature_row[name] for name in feature_order]],
        dtype=float,
    )