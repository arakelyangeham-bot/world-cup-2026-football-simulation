#validate_club_goal_model_baseline

from __future__ import annotations

from research.baselines.club_goal_model import (
    CLUB_GOAL_MODEL_V1,
    CURRENT_CLUB_GOAL_MODEL,
    get_club_goal_model_baseline,
    list_club_goal_model_baselines,
)


EXPECTED_VERSION = "1.0"

EXPECTED_FEATURE_SPECIFICATION = (
    "attack_defense_attack_depth_rating_prior"
)

EXPECTED_SUPPORTING_STUDIES = (
    "052",
    "054",
    "060",
    "061",
)

EXPECTED_HOME_FEATURES = (
    "home_attack",
    "away_defense",
    "attack_depth_diff",
    "rating_prior_diff",
)

EXPECTED_AWAY_FEATURES = (
    "away_attack",
    "home_defense",
    "attack_depth_diff",
    "rating_prior_diff",
)


def validate_registry() -> None:
    versions = (
        list_club_goal_model_baselines()
    )

    if EXPECTED_VERSION not in versions:
        raise AssertionError(
            "Version 1.0 is missing from the club "
            "goal-model baseline registry."
        )

    retrieved = (
        get_club_goal_model_baseline(
            EXPECTED_VERSION
        )
    )

    if retrieved is not CLUB_GOAL_MODEL_V1:
        raise AssertionError(
            "Registry lookup did not return the "
            "Version 1 baseline instance."
        )


def validate_current_baseline() -> None:
    if (
        CURRENT_CLUB_GOAL_MODEL
        is not CLUB_GOAL_MODEL_V1
    ):
        raise AssertionError(
            "The current club goal-model baseline is "
            "not Version 1."
        )

    if (
        CURRENT_CLUB_GOAL_MODEL.status
        != "recommended"
    ):
        raise AssertionError(
            "The current baseline is not marked "
            "recommended."
        )


def validate_evidence_record() -> None:
    if (
        CLUB_GOAL_MODEL_V1.supporting_studies
        != EXPECTED_SUPPORTING_STUDIES
    ):
        raise AssertionError(
            "Unexpected supporting-study sequence."
        )


def validate_feature_specification() -> None:
    baseline = CLUB_GOAL_MODEL_V1

    if (
        baseline.feature_specification
        != EXPECTED_FEATURE_SPECIFICATION
    ):
        raise AssertionError(
            "Unexpected Version 1 feature "
            "specification."
        )

    specification = (
        baseline.get_feature_specification()
    )

    if (
        specification.home_features
        != EXPECTED_HOME_FEATURES
    ):
        raise AssertionError(
            "Unexpected Version 1 home-goal "
            "features."
        )

    if (
        specification.away_features
        != EXPECTED_AWAY_FEATURES
    ):
        raise AssertionError(
            "Unexpected Version 1 away-goal "
            "features."
        )

    if (
        "rating_prior_diff"
        not in specification.required_columns()
    ):
        raise AssertionError(
            "Version 1 does not include the historical "
            "rating prior."
        )

    if (
        "attack_depth_diff"
        not in specification.required_columns()
    ):
        raise AssertionError(
            "Version 1 does not include attacking depth."
        )


def main() -> None:
    CLUB_GOAL_MODEL_V1.validate()

    validate_registry()
    validate_current_baseline()
    validate_evidence_record()
    validate_feature_specification()

    specification = (
        CLUB_GOAL_MODEL_V1
        .get_feature_specification()
    )

    print(
        "Club Goal Model Baseline Validation"
    )
    print("=" * 76)
    print()
    print(
        f"Name: {CLUB_GOAL_MODEL_V1.name}"
    )
    print(
        f"Version: {CLUB_GOAL_MODEL_V1.version}"
    )
    print(
        f"Status: {CLUB_GOAL_MODEL_V1.status}"
    )
    print(
        "Feature specification: "
        f"{CLUB_GOAL_MODEL_V1.feature_specification}"
    )
    print(
        "Supporting studies: "
        f"{', '.join(CLUB_GOAL_MODEL_V1.supporting_studies)}"
    )
    print()
    print(
        "Home-goal features: "
        f"{', '.join(specification.home_features)}"
    )
    print(
        "Away-goal features: "
        f"{', '.join(specification.away_features)}"
    )
    print()
    print("Baseline registry: PASS")
    print("Current-baseline pointer: PASS")
    print("Evidence record: PASS")
    print("Feature-registry integration: PASS")
    print("Home-feature contract: PASS")
    print("Away-feature contract: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()