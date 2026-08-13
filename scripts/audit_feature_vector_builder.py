#audit_feature_vector_builder.py

import numpy as np

from scripts.team_strength_loader import load_team_repository
from inference.match_predictor import MatchPredictor
from inference.feature_builder import build_engineered_features
from inference.feature_vector_builder import (
    get_feature_order,
    build_feature_vector,
)


def main() -> None:
    repo = load_team_repository()
    predictor = MatchPredictor()

    home = repo["Argentina"]
    away = repo["France"]

    feature_order = get_feature_order(predictor.model)

    feature_row = build_engineered_features(home, away)
    vector = build_feature_vector(home, away, feature_order)

    print("Feature vector audit")
    print("--------------------")
    print(f"Features: {len(feature_order)}")
    print(f"Vector shape: {vector.shape}")

    print()
    print("Feature order check")
    print("-------------------")

    mismatches = []

    for idx, name in enumerate(feature_order):
        expected = feature_row[name]
        actual = vector[0, idx]

        if not np.isclose(expected, actual):
            mismatches.append((name, expected, actual))

        print(f"{idx:2} {name:<28} {actual: .6f}")

    print()
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        for name, expected, actual in mismatches:
            print(f"{name}: expected={expected}, actual={actual}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()