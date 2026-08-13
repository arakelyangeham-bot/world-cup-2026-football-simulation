#test_bundesliga_production_goal_prediction

from __future__ import annotations

import math
from pathlib import Path

from simulation.production_goal_model import (
    ProductionGoalModel,
    ProductionGoalPrediction,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

# Adjust only if these modules live at different paths.
from simulation.live_match_observation_builder import (
    LiveMatchObservationBuilder,
    ProductionClubRepository,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)

from simulation.match_engine_adapter import (
    simulate_scoreline_from_lambdas,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BUNDESLIGA_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_078_bundesliga_production_repository"
    / "bundesliga_club_repository_v1.csv"
)

GOAL_MODEL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)


HOME_TEAM = "FC Bayern München"
AWAY_TEAM = "Borussia Dortmund"
PREDICTION_DATE = "2025-02-01"

EXPECTED_ARTIFACT_NAME = (
    "integrated_club_goal_model"
)
EXPECTED_ARTIFACT_VERSION = "1.0"
EXPECTED_BASELINE_VERSION = "1.0"

EXPECTED_FEATURES = {
    "home_attack",
    "away_attack",
    "home_defense",
    "away_defense",
    "attack_depth_diff",
    "rating_prior_diff",
}


def validate_prediction(
    prediction: ProductionGoalPrediction,
) -> None:
    values = {
        "lambda_home": prediction.lambda_home,
        "lambda_away": prediction.lambda_away,
        "pred_total_goals": (
            prediction.pred_total_goals
        ),
        "pred_goal_diff": (
            prediction.pred_goal_diff
        ),
    }

    for name, value in values.items():
        if not math.isfinite(value):
            raise AssertionError(
                "Prediction contains a non-finite value. "
                f"Field={name!r}, value={value!r}."
            )

    if prediction.lambda_home <= 0.0:
        raise AssertionError(
            "Home expected goals must be positive."
        )

    if prediction.lambda_away <= 0.0:
        raise AssertionError(
            "Away expected goals must be positive."
        )

    expected_total = (
        prediction.lambda_home
        + prediction.lambda_away
    )

    if not math.isclose(
        prediction.pred_total_goals,
        expected_total,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "Predicted total goals are inconsistent."
        )

    expected_difference = (
        prediction.lambda_home
        - prediction.lambda_away
    )

    if not math.isclose(
        prediction.pred_goal_diff,
        expected_difference,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "Predicted goal difference is inconsistent."
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 080 — BUNDESLIGA PRODUCTION "
        "GOAL PREDICTION"
    )
    print("=" * 88)

    if not GOAL_MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Production goal-model artifact does not exist: "
            f"{GOAL_MODEL_ARTIFACT_PATH}"
        )

    club_repository = ProductionClubRepository(
        repository_path=(
            BUNDESLIGA_REPOSITORY_PATH
        )
    )

    clubelo_repository = ClubEloRepository(
        cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        )
    )

    observation_builder = LiveMatchObservationBuilder(
        club_repository=club_repository,
        clubelo_repository=clubelo_repository,
        clubelo_name_overrides=(
            BUNDESLIGA_CLUBELO_NAME_OVERRIDES
        ),
    )

    goal_model = ProductionGoalModel.from_path(
        GOAL_MODEL_ARTIFACT_PATH
    )

    print()
    print("Runtime configuration")
    print(
        "  Production repository: "
        f"{BUNDESLIGA_REPOSITORY_PATH}"
    )
    print(
        "  Goal-model artifact: "
        f"{GOAL_MODEL_ARTIFACT_PATH}"
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

    if (
        goal_model.artifact_name
        != EXPECTED_ARTIFACT_NAME
    ):
        raise AssertionError(
            "Unexpected artifact name. "
            f"Expected {EXPECTED_ARTIFACT_NAME!r}, "
            f"received {goal_model.artifact_name!r}."
        )

    if (
        goal_model.artifact_version
        != EXPECTED_ARTIFACT_VERSION
    ):
        raise AssertionError(
            "Unexpected artifact version."
        )

    if (
        goal_model.baseline_version
        != EXPECTED_BASELINE_VERSION
    ):
        raise AssertionError(
            "Unexpected baseline version."
        )

    print()
    print("Constructing live observation...")

    observation = observation_builder.build(
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        prediction_date=PREDICTION_DATE,
    )

    feature_mapping = (
        observation.to_feature_mapping()
    )

    if set(feature_mapping) != EXPECTED_FEATURES:
        raise AssertionError(
            "Live observation feature set differs from "
            "the expected production contract."
        )

    if set(goal_model.required_features) != (
        EXPECTED_FEATURES
    ):
        raise AssertionError(
            "Frozen goal-model artifact requires an "
            "unexpected feature set. "
            f"Required: "
            f"{goal_model.required_features}"
        )

    print()
    print("Generating expected-goal prediction...")

    prediction = goal_model.predict(
        feature_mapping
    )

    repeated_prediction = goal_model.predict(
        feature_mapping
    )

    validate_prediction(prediction)
    validate_prediction(repeated_prediction)

    if not math.isclose(
        prediction.lambda_home,
        repeated_prediction.lambda_home,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AssertionError(
            "Repeated home-lambda prediction changed."
        )

    if not math.isclose(
        prediction.lambda_away,
        repeated_prediction.lambda_away,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AssertionError(
            "Repeated away-lambda prediction changed."
        )

    print()
    print("Model provenance")
    print(
        f"  Artifact name: "
        f"{goal_model.artifact_name}"
    )
    print(
        f"  Artifact version: "
        f"{goal_model.artifact_version}"
    )
    print(
        f"  Baseline name: "
        f"{goal_model.baseline_name}"
    )
    print(
        f"  Baseline version: "
        f"{goal_model.baseline_version}"
    )
    print(
        "  Feature specification: "
        f"{goal_model.feature_specification}"
    )
    print(
        "  Training cutoff: "
        f"{goal_model.training_end_date}"
    )

    print()
    print("Feature mapping")

    for feature_name in sorted(
        feature_mapping
    ):
        print(
            f"  {feature_name:<24} "
            f"{feature_mapping[feature_name]:.6f}"
        )

    print()
    print("Expected-goal prediction")
    print(
        f"  Lambda home: "
        f"{prediction.lambda_home:.6f}"
    )
    print(
        f"  Lambda away: "
        f"{prediction.lambda_away:.6f}"
    )
    print(
        f"  Predicted total goals: "
        f"{prediction.pred_total_goals:.6f}"
    )
    print(
        f"  Predicted goal difference: "
        f"{prediction.pred_goal_diff:.6f}"
    )

    # Interface validation only. This confirms that the
    # downstream scoreline sampler accepts the predicted lambdas.
    sampled_score = simulate_scoreline_from_lambdas(
        lambda_home=prediction.lambda_home,
        lambda_away=prediction.lambda_away,
    )

    if (
        not isinstance(sampled_score, tuple)
        or len(sampled_score) != 2
        or not all(
            isinstance(value, int)
            for value in sampled_score
        )
        or not all(
            value >= 0
            for value in sampled_score
        )
    ):
        raise AssertionError(
            "Scoreline sampler returned an invalid result: "
            f"{sampled_score!r}"
        )

    print()
    print("Scoreline interface")
    print(
        f"  Sampled scoreline: "
        f"{HOME_TEAM} "
        f"{sampled_score[0]}–{sampled_score[1]} "
        f"{AWAY_TEAM}"
    )

    print()
    print("Validation summary")
    print("  Artifact loading: PASS")
    print("  Artifact provenance: PASS")
    print("  Observation construction: PASS")
    print("  Feature compatibility: PASS")
    print("  Positive finite lambdas: PASS")
    print("  Prediction arithmetic: PASS")
    print("  Deterministic prediction: PASS")
    print("  Scoreline sampler interface: PASS")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()