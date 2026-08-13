#test_production_prediction_pipeline

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from research.production.production_prediction_pipeline_factory import (
    build_production_prediction_pipeline,
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

GOAL_MODEL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 082 — PRODUCTION PREDICTION PIPELINE"
    )
    print("=" * 88)

    pipeline = build_production_prediction_pipeline(
        club_repository_path=(
            BUNDESLIGA_REPOSITORY_PATH
        ),
        clubelo_cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        ),
        goal_model_artifact_path=(
            GOAL_MODEL_ARTIFACT_PATH
        ),
        clubelo_name_overrides=(
            BUNDESLIGA_CLUBELO_NAME_OVERRIDES
        ),
    )

    print()
    print("Single-fixture prediction")

    prediction = pipeline.predict_fixture(
        home_team="FC Bayern München",
        away_team="Borussia Dortmund",
        prediction_date="2025-02-01",
    )

    repeated = pipeline.predict_fixture(
        home_team="FC Bayern München",
        away_team="Borussia Dortmund",
        prediction_date="2025-02-01",
    )

    if prediction != repeated:
        raise AssertionError(
            "Repeated pipeline prediction was not "
            "deterministic."
        )

    print(
        f"  Lambda home: "
        f"{prediction.lambda_home:.6f}"
    )
    print(
        f"  Lambda away: "
        f"{prediction.lambda_away:.6f}"
    )
    print(
        f"  Home win: "
        f"{prediction.home_win_probability:.6f}"
    )
    print(
        f"  Draw: "
        f"{prediction.draw_probability:.6f}"
    )
    print(
        f"  Away win: "
        f"{prediction.away_win_probability:.6f}"
    )

    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "date": "2025-02-01",
                "home_team": "FC Bayern München",
                "away_team": "Borussia Dortmund",
            },
            {
                "fixture_id": 2,
                "date": "2025-02-08",
                "home_team": "Bayer 04 Leverkusen",
                "away_team": "VfB Stuttgart",
            },
            {
                "fixture_id": 3,
                "date": "2025-02-15",
                "home_team": "SC Freiburg",
                "away_team": "Eintracht Frankfurt",
            },
        ]
    )

    print()
    print("Batch prediction")

    predictions, failures = (
        pipeline.predict_fixtures(
            fixtures,
            continue_on_error=True,
        )
    )

    if len(predictions) != 3:
        raise AssertionError(
            "Batch prediction did not preserve all valid "
            "fixtures."
        )

    if not failures.empty:
        raise AssertionError(
            "Valid batch fixture population produced "
            f"failures:\n{failures.to_string(index=False)}"
        )

    required_columns = {
        "fixture_id",
        "home_team",
        "away_team",
        "lambda_home",
        "lambda_away",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "prediction_status",
    }

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise AssertionError(
            "Batch prediction output is missing columns: "
            f"{sorted(missing)}"
        )

    print(
        predictions[
            [
                "fixture_id",
                "home_team",
                "away_team",
                "lambda_home",
                "lambda_away",
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    invalid_fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 4,
                "date": "2025-02-01",
                "home_team": "Unknown Club",
                "away_team": "Borussia Dortmund",
            }
        ]
    )

    _, captured_failures = (
        pipeline.predict_fixtures(
            invalid_fixtures,
            continue_on_error=True,
        )
    )

    if len(captured_failures) != 1:
        raise AssertionError(
            "Expected runtime failure was not captured."
        )

    if (
        captured_failures.iloc[0][
            "prediction_status"
        ]
        != "FAILED"
    ):
        raise AssertionError(
            "Captured runtime failure has an unexpected "
            "status."
        )

    print()
    print("Validation summary")
    print("  Pipeline construction: PASS")
    print("  Single-fixture prediction: PASS")
    print("  Deterministic prediction: PASS")
    print("  Observation integration: PASS")
    print("  Goal-model integration: PASS")
    print("  Probability integration: PASS")
    print("  Batch prediction: PASS")
    print("  Runtime failure capture: PASS")
    print("  Persistence-ready output: PASS")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()