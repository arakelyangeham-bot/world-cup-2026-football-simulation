#test_bundesliga_live_observation

from __future__ import annotations

import math
from pathlib import Path

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

# Adjust this import only if the module lives elsewhere.
from simulation.live_match_observation_builder import (
    LiveMatchObservation,
    LiveMatchObservationBuilder,
    ProductionClubRepository,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BUNDESLIGA_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_078_bundesliga_production_repository"
    / "bundesliga_club_repository_v1.csv"
)


HOME_TEAM = "FC Bayern München"
AWAY_TEAM = "Borussia Dortmund"

# This date falls within the 2024–25 Bundesliga season and should
# be covered by both cached ClubElo histories.
PREDICTION_DATE = "2025-02-01"


EXPECTED_FEATURES = {
    "home_attack",
    "away_attack",
    "home_defense",
    "away_defense",
    "attack_depth_diff",
    "rating_prior_diff",
}


def validate_observation(
    observation: LiveMatchObservation,
) -> dict[str, float]:
    """
    Validate the complete runtime observation and return the exact
    production model feature mapping.
    """

    if observation.home_team == observation.away_team:
        raise AssertionError(
            "Home and away teams resolved to the same club."
        )

    if observation.home_team != HOME_TEAM:
        raise AssertionError(
            "Unexpected resolved home team. "
            f"Expected {HOME_TEAM!r}, "
            f"received {observation.home_team!r}."
        )

    if observation.away_team != AWAY_TEAM:
        raise AssertionError(
            "Unexpected resolved away team. "
            f"Expected {AWAY_TEAM!r}, "
            f"received {observation.away_team!r}."
        )

    if str(observation.prediction_date) != PREDICTION_DATE:
        raise AssertionError(
            "Prediction date changed during observation "
            "construction."
        )

    if not (
        observation.home_rating_effective_from
        <= observation.prediction_date
        <= observation.home_rating_effective_to
    ):
        raise AssertionError(
            "Home ClubElo interval is not valid on the "
            "prediction date."
        )

    if not (
        observation.away_rating_effective_from
        <= observation.prediction_date
        <= observation.away_rating_effective_to
    ):
        raise AssertionError(
            "Away ClubElo interval is not valid on the "
            "prediction date."
        )

    if observation.rating_prior_source != "clubelo":
        raise AssertionError(
            "Unexpected rating-prior source: "
            f"{observation.rating_prior_source!r}"
        )

    numeric_values = {
        "home_attack": observation.home_attack,
        "away_attack": observation.away_attack,
        "home_defense": observation.home_defense,
        "away_defense": observation.away_defense,
        "home_attack_depth": (
            observation.home_attack_depth
        ),
        "away_attack_depth": (
            observation.away_attack_depth
        ),
        "attack_depth_diff": (
            observation.attack_depth_diff
        ),
        "home_rating_prior": (
            observation.home_rating_prior
        ),
        "away_rating_prior": (
            observation.away_rating_prior
        ),
        "rating_prior_diff": (
            observation.rating_prior_diff
        ),
    }

    for field_name, value in numeric_values.items():
        if not math.isfinite(value):
            raise AssertionError(
                "Observation contains a non-finite numeric "
                f"value. Field={field_name!r}, value={value!r}"
            )

    expected_attack_depth_diff = (
        observation.home_attack_depth
        - observation.away_attack_depth
    )

    if not math.isclose(
        observation.attack_depth_diff,
        expected_attack_depth_diff,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "attack_depth_diff is inconsistent with the "
            "underlying home and away values."
        )

    expected_rating_prior_diff = (
        observation.home_rating_prior
        - observation.away_rating_prior
    )

    if not math.isclose(
        observation.rating_prior_diff,
        expected_rating_prior_diff,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "rating_prior_diff is inconsistent with the "
            "underlying ClubElo ratings."
        )

    feature_mapping = (
        observation.to_feature_mapping()
    )

    if set(feature_mapping) != EXPECTED_FEATURES:
        raise AssertionError(
            "Observation feature mapping does not match the "
            "Integrated Club Goal Model contract. "
            f"Expected {sorted(EXPECTED_FEATURES)}, "
            f"received {sorted(feature_mapping)}."
        )

    for feature_name, value in (
        feature_mapping.items()
    ):
        if not math.isfinite(value):
            raise AssertionError(
                "Feature mapping contains a non-finite "
                f"value. Feature={feature_name!r}, "
                f"value={value!r}."
            )

    return feature_mapping


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 079B — BUNDESLIGA LIVE "
        "OBSERVATION INTEGRATION"
    )
    print("=" * 88)

    production_repository = (
        ProductionClubRepository(
            repository_path=(
                BUNDESLIGA_REPOSITORY_PATH
            )
        )
    )

    clubelo_repository = ClubEloRepository(
        cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        )
    )

    observation_builder = (
        LiveMatchObservationBuilder(
            club_repository=(
                production_repository
            ),
            clubelo_repository=(
                clubelo_repository
            ),
            clubelo_name_overrides=(
                BUNDESLIGA_CLUBELO_NAME_OVERRIDES
            ),
        )
    )

    print()
    print("Runtime inputs")
    print(
        "  Production repository: "
        f"{BUNDESLIGA_REPOSITORY_PATH}"
    )
    print(
        "  ClubElo cache: "
        f"{CLUBELO_CACHE_DIRECTORY}"
    )
    print(
        f"  Home team: {HOME_TEAM}"
    )
    print(
        f"  Away team: {AWAY_TEAM}"
    )
    print(
        f"  Prediction date: {PREDICTION_DATE}"
    )

    print()
    print("Constructing observation...")

    observation = observation_builder.build(
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        prediction_date=PREDICTION_DATE,
    )

    feature_mapping = validate_observation(
        observation
    )

    print()
    print("Resolved match")
    print(
        "  Requested home team: "
        f"{observation.requested_home_team}"
    )
    print(
        "  Requested away team: "
        f"{observation.requested_away_team}"
    )
    print(
        f"  Resolved home team: {observation.home_team}"
    )
    print(
        f"  Resolved away team: {observation.away_team}"
    )

    print()
    print("Football intelligence")
    print(
        f"  Home attack: {observation.home_attack:.6f}"
    )
    print(
        f"  Away attack: {observation.away_attack:.6f}"
    )
    print(
        f"  Home defense: {observation.home_defense:.6f}"
    )
    print(
        f"  Away defense: {observation.away_defense:.6f}"
    )
    print(
        "  Home attack depth: "
        f"{observation.home_attack_depth:.6f}"
    )
    print(
        "  Away attack depth: "
        f"{observation.away_attack_depth:.6f}"
    )
    print(
        "  Attack-depth difference: "
        f"{observation.attack_depth_diff:.6f}"
    )

    print()
    print("ClubElo rating priors")
    print(
        "  Home rating: "
        f"{observation.home_rating_prior:.3f}"
    )
    print(
        "  Away rating: "
        f"{observation.away_rating_prior:.3f}"
    )
    print(
        "  Rating-prior difference: "
        f"{observation.rating_prior_diff:.3f}"
    )
    print(
        "  Home rating interval: "
        f"{observation.home_rating_effective_from} "
        "through "
        f"{observation.home_rating_effective_to}"
    )
    print(
        "  Away rating interval: "
        f"{observation.away_rating_effective_from} "
        "through "
        f"{observation.away_rating_effective_to}"
    )

    print()
    print("Production feature mapping")

    for feature_name in sorted(
        feature_mapping
    ):
        print(
            f"  {feature_name:<24} "
            f"{feature_mapping[feature_name]:.6f}"
        )

    print()
    print("Validation summary")
    print("  Home club resolution: PASS")
    print("  Away club resolution: PASS")
    print("  Home ClubElo resolution: PASS")
    print("  Away ClubElo resolution: PASS")
    print("  Home temporal validity: PASS")
    print("  Away temporal validity: PASS")
    print("  Observation construction: PASS")
    print("  Derived differences: PASS")
    print("  Model feature contract: PASS")
    print("  Finite feature values: PASS")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()