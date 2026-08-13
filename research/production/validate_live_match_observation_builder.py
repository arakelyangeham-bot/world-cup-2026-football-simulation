#validate_live_match_observation_builder

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.baselines.club_goal_model import (
    CURRENT_CLUB_GOAL_MODEL,
)
from simulation.integrated_club_goal_predictor import (
    IntegratedClubGoalPredictor,
)
from simulation.live_match_observation_builder import (
    DEFAULT_CLUB_REPOSITORY_PATH,
    LiveMatchObservationBuilder,
    ProductionClubRepository,
)

# Adjust this import to the same package used in the runtime
# builder if necessary.
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_071_live_match_observation_builder"
)

VALIDATION_PATH = (
    OUTPUT_DIRECTORY
    / "live_match_observation_validation.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


# Update this only if your historical ClubElo cache lives
# elsewhere.
CLUBELO_CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "clubelo_histories"
)


VALIDATION_HOME_TEAM = "Arsenal"
VALIDATION_AWAY_TEAM = "Everton"

# First date after the production artifact's training cutoff.
VALIDATION_PREDICTION_DATE = "2025-05-26"


def build_runtime_components():
    club_repository = (
        ProductionClubRepository(
            DEFAULT_CLUB_REPOSITORY_PATH
        )
    )

    clubelo_repository = (
        ClubEloRepository(
            cache_directory=(
                CLUBELO_CACHE_DIRECTORY
            )
        )
    )

    builder = LiveMatchObservationBuilder(
        club_repository=club_repository,
        clubelo_repository=(
            clubelo_repository
        ),
    )

    predictor = (
        IntegratedClubGoalPredictor()
    )

    return (
        club_repository,
        builder,
        predictor,
    )


def validate_live_observation():
    (
        club_repository,
        builder,
        predictor,
    ) = build_runtime_components()

    observation = builder.build(
        home_team=VALIDATION_HOME_TEAM,
        away_team=VALIDATION_AWAY_TEAM,
        prediction_date=(
            VALIDATION_PREDICTION_DATE
        ),
    )

    home = club_repository.resolve_club(
        VALIDATION_HOME_TEAM
    )

    away = club_repository.resolve_club(
        VALIDATION_AWAY_TEAM
    )

    expected_attack_depth_diff = (
        home.attack_depth
        - away.attack_depth
    )

    if not np.isclose(
        observation.attack_depth_diff,
        expected_attack_depth_diff,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "attack_depth_diff arithmetic failed."
        )

    expected_rating_prior_diff = (
        observation.home_rating_prior
        - observation.away_rating_prior
    )

    if not np.isclose(
        observation.rating_prior_diff,
        expected_rating_prior_diff,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "rating_prior_diff arithmetic failed."
        )

    prediction_date = (
        observation.prediction_date
    )

    if not (
        observation.home_rating_effective_from
        <= prediction_date
        <= observation.home_rating_effective_to
    ):
        raise AssertionError(
            "Home ClubElo interval is not valid on the "
            "prediction date."
        )

    if not (
        observation.away_rating_effective_from
        <= prediction_date
        <= observation.away_rating_effective_to
    ):
        raise AssertionError(
            "Away ClubElo interval is not valid on the "
            "prediction date."
        )

    feature_values = (
        observation.to_feature_mapping()
    )

    specification = (
        CURRENT_CLUB_GOAL_MODEL
        .get_feature_specification()
    )

    if set(feature_values) != set(
        specification.required_columns()
    ):
        raise AssertionError(
            "Live feature mapping differs from the "
            "registered model contract."
        )

    prediction = predictor.predict_features(
        feature_values=feature_values,
        prediction_date=prediction_date,
    )

    numeric_values = np.asarray(
        [
            *feature_values.values(),
            prediction.lambda_home,
            prediction.lambda_away,
        ],
        dtype=float,
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            "Live observation or prediction contains "
            "non-finite values."
        )

    if (
        prediction.lambda_home <= 0.0
        or prediction.lambda_away <= 0.0
    ):
        raise AssertionError(
            "Production predictor returned non-positive "
            "expected goals."
        )

    record = {
        "requested_home_team":
            observation.requested_home_team,
        "requested_away_team":
            observation.requested_away_team,
        "home_team":
            observation.home_team,
        "away_team":
            observation.away_team,
        "prediction_date":
            observation.prediction_date.isoformat(),
        "home_attack":
            observation.home_attack,
        "away_attack":
            observation.away_attack,
        "home_defense":
            observation.home_defense,
        "away_defense":
            observation.away_defense,
        "home_attack_depth":
            observation.home_attack_depth,
        "away_attack_depth":
            observation.away_attack_depth,
        "attack_depth_diff":
            observation.attack_depth_diff,
        "home_rating_prior":
            observation.home_rating_prior,
        "away_rating_prior":
            observation.away_rating_prior,
        "rating_prior_diff":
            observation.rating_prior_diff,
        "home_rating_effective_from":
            observation.home_rating_effective_from.isoformat(),
        "home_rating_effective_to":
            observation.home_rating_effective_to.isoformat(),
        "away_rating_effective_from":
            observation.away_rating_effective_from.isoformat(),
        "away_rating_effective_to":
            observation.away_rating_effective_to.isoformat(),
        "rating_prior_source":
            observation.rating_prior_source,
        "repository_version":
            observation.repository_version,
        "repository_scope":
            observation.repository_scope,
        "lambda_home":
            prediction.lambda_home,
        "lambda_away":
            prediction.lambda_away,
        "artifact_name":
            prediction.artifact_name,
        "artifact_version":
            prediction.artifact_version,
        "feature_specification":
            prediction.feature_specification,
    }

    return record


def validate_unknown_club_rejection() -> None:
    _, builder, _ = (
        build_runtime_components()
    )

    try:
        builder.build(
            home_team=(
                "Definitely Not A Real Club"
            ),
            away_team=(
                VALIDATION_AWAY_TEAM
            ),
            prediction_date=(
                VALIDATION_PREDICTION_DATE
            ),
        )
    except KeyError:
        return

    raise AssertionError(
        "Builder accepted a club absent from the "
        "production repository."
    )


def validate_same_club_rejection() -> None:
    _, builder, _ = (
        build_runtime_components()
    )

    try:
        builder.build(
            home_team=(
                VALIDATION_HOME_TEAM
            ),
            away_team=(
                VALIDATION_HOME_TEAM
            ),
            prediction_date=(
                VALIDATION_PREDICTION_DATE
            ),
        )
    except ValueError:
        return

    raise AssertionError(
        "Builder accepted the same home and away club."
    )


def build_metadata(
    record: dict[str, object],
) -> dict[str, object]:
    return {
        "study_id": "071",
        "study_name": (
            "Live Match Observation Builder"
        ),
        "repository_path": str(
            DEFAULT_CLUB_REPOSITORY_PATH
        ),
        "clubelo_cache_directory": str(
            CLUBELO_CACHE_DIRECTORY
        ),
        "validation_home_team":
            record["home_team"],
        "validation_away_team":
            record["away_team"],
        "validation_prediction_date":
            record["prediction_date"],
        "required_features": list(
            CURRENT_CLUB_GOAL_MODEL
            .get_feature_specification()
            .required_columns()
        ),
        "lambda_home":
            record["lambda_home"],
        "lambda_away":
            record["lambda_away"],
        "repository_loading_pass": True,
        "club_resolution_pass": True,
        "clubelo_resolution_pass": True,
        "temporal_validity_pass": True,
        "attack_depth_arithmetic_pass": True,
        "rating_prior_arithmetic_pass": True,
        "feature_contract_pass": True,
        "predictor_compatibility_pass": True,
        "unknown_club_rejection_pass": True,
        "same_club_rejection_pass": True,
        "overall_result": "PASS",
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 071 — Live Match Observation Builder

    ## Purpose

    Construct the exact Integrated Club Goal Model v1 feature
    mapping from a home club, away club, and prediction date.

    ## Runtime architecture

    ```text
    Production Club Repository v1
            +
    Prediction-date ClubElo repository
            ↓
    LiveMatchObservationBuilder
            ↓
    Version 1 feature mapping
            ↓
    IntegratedClubGoalPredictor
            ↓
    Expected goals

    Validation match
    Home: {metadata["validation_home_team"]}
    Away: {metadata["validation_away_team"]}
    Prediction date:
    {metadata["validation_prediction_date"]}
    Required features

    {chr(10).join(
    f"- {feature}"
    for feature in metadata["required_features"]
    )}

    Prediction
    Expected home goals:
    {metadata["lambda_home"]:.6f}
    Expected away goals:
    {metadata["lambda_away"]:.6f}
    Validation
    Production repository loading: PASS
    Club-name resolution: PASS
    Prediction-date ClubElo resolution: PASS
    ClubElo temporal validity: PASS
    Attack-depth difference arithmetic: PASS
    Rating-prior difference arithmetic: PASS
    Baseline Registry integration: PASS
    Feature Registry integration: PASS
    Production predictor compatibility: PASS
    Unknown-club rejection: PASS
    Same-club rejection: PASS
    Finite expected goals: PASS
    Positive expected goals: PASS
    Boundary established

    The live builder performs no fitting and does not load
    historical match observations. It only orchestrates validated
    production providers and constructs the registered Version 1
    feature contract.

    Result

    OVERALL RESULT: PASS
    """

def main() -> None:
    record = validate_live_observation()

    validate_unknown_club_rejection()
    validate_same_club_rejection()

    metadata = build_metadata(
        record
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [record]
    ).to_csv(
        VALIDATION_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(metadata)

    print(
        "Study 071 — Live Match Observation Builder"
    )
    print("=" * 76)
    print()
    print(
        "Validation match: "
        f"{record['home_team']} vs "
        f"{record['away_team']}"
    )
    print(
        "Prediction date: "
        f"{record['prediction_date']}"
    )
    print()
    print("Live Features")
    print("-" * 76)
    print(
        f"home_attack: "
        f"{record['home_attack']:.10f}"
    )
    print(
        f"away_attack: "
        f"{record['away_attack']:.10f}"
    )
    print(
        f"home_defense: "
        f"{record['home_defense']:.10f}"
    )
    print(
        f"away_defense: "
        f"{record['away_defense']:.10f}"
    )
    print(
        f"attack_depth_diff: "
        f"{record['attack_depth_diff']:.10f}"
    )
    print(
        f"rating_prior_diff: "
        f"{record['rating_prior_diff']:.10f}"
    )
    print()
    print("Production Prediction")
    print("-" * 76)
    print(
        f"lambda_home: "
        f"{record['lambda_home']:.10f}"
    )
    print(
        f"lambda_away: "
        f"{record['lambda_away']:.10f}"
    )
    print()
    print("Repository loading: PASS")
    print("Club resolution: PASS")
    print("ClubElo resolution: PASS")
    print("Temporal validity: PASS")
    print("Attack-depth arithmetic: PASS")
    print("Rating-prior arithmetic: PASS")
    print("Feature contract: PASS")
    print("Predictor compatibility: PASS")
    print("Unknown-club rejection: PASS")
    print("Same-club rejection: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )

if __name__ == "__main__":
    main()