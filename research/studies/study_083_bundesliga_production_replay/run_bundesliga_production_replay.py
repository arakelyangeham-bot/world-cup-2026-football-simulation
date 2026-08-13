#run_bundesliga_production_replay

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from research.production.production_prediction_pipeline_factory import (
    build_production_prediction_pipeline,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
    / "bundesliga"
    / "bundesliga_2024_completed_matches.csv"
)

DEFAULT_BUNDESLIGA_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_078_bundesliga_production_repository"
    / "bundesliga_club_repository_v1.csv"
)

DEFAULT_GOAL_MODEL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_083_bundesliga_production_replay"
)

DEFAULT_STUDY_ID = "083"

DEFAULT_STUDY_NAME = (
    "Bundesliga Production Replay Validation"
)


EXPECTED_MATCH_COUNT = 306
EXPECTED_CLUB_COUNT = 18
EXPECTED_SEASON_START_YEAR = 2024
EXPECTED_COMPETITION_KEY = "bundesliga"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Bundesliga historical fixture population "
            "through a persisted production repository and "
            "goal-model artifact."
        )
    )

    parser.add_argument(
        "--repository-path",
        type=Path,
        default=(
            DEFAULT_BUNDESLIGA_REPOSITORY_PATH
        ),
        help=(
            "Production-format Bundesliga club repository."
        ),
    )

    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=(
            DEFAULT_GOAL_MODEL_ARTIFACT_PATH
        ),
        help=(
            "Serialized production club goal-model artifact."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory receiving replay predictions, "
            "failures, summary, and metadata."
        ),
    )

    parser.add_argument(
        "--study-id",
        default=DEFAULT_STUDY_ID,
        help="Study identifier written to metadata.",
    )

    parser.add_argument(
        "--study-name",
        default=DEFAULT_STUDY_NAME,
        help="Study name written to metadata.",
    )

    parser.add_argument(
        "--candidate-name",
        default="canonical",
        help=(
            "Candidate label written to metadata."
        ),
    )

    return parser.parse_args()

def load_replay_population() -> pd.DataFrame:
    """
    Load and validate the canonical Bundesliga 2024–25
    completed-match population.

    This function validates operational suitability only.
    It does not make any claim about evaluation independence
    or absence of information leakage.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Bundesliga replay input does not exist: "
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Bundesliga replay input is empty."
        )

    required_columns = {
        "competition_key",
        "season_start_year",
        "event_id",
        "date",
        "stage",
        "round",
        "round_number",
        "home_team",
        "home_team_id",
        "away_team",
        "away_team_id",
        "home_score",
        "away_score",
        "goal_difference",
        "total_goals",
        "outcome",
        "status_code",
        "status_desc",
        "winner",
        "completed",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Bundesliga replay input is missing required "
            f"columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()

    if len(dataframe) != EXPECTED_MATCH_COUNT:
        raise ValueError(
            "Unexpected Bundesliga replay population. "
            f"Expected {EXPECTED_MATCH_COUNT} matches, "
            f"received {len(dataframe)}."
        )

    if dataframe["event_id"].duplicated().any():
        duplicates = (
            dataframe.loc[
                dataframe["event_id"].duplicated(
                    keep=False
                ),
                [
                    "event_id",
                    "home_team",
                    "away_team",
                ],
            ]
            .sort_values("event_id")
        )

        raise ValueError(
            "Bundesliga replay input contains duplicate "
            f"event IDs:\n{duplicates.to_string(index=False)}"
        )

    if not dataframe[
        "competition_key"
    ].eq(
        EXPECTED_COMPETITION_KEY
    ).all():
        raise ValueError(
            "Replay input contains an unexpected "
            "competition key."
        )

    if not dataframe[
        "season_start_year"
    ].eq(
        EXPECTED_SEASON_START_YEAR
    ).all():
        raise ValueError(
            "Replay input contains an unexpected season."
        )

    completed = (
        dataframe["completed"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if completed.isna().any():
        raise ValueError(
            "Replay input contains unrecognized completion "
            "values."
        )

    if not completed.all():
        raise ValueError(
            "Replay input contains incomplete fixtures."
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    # LiveMatchObservationBuilder accepts calendar dates.
    # Preserve the original timestamp while supplying a clean
    # date to the prediction pipeline.
    dataframe["fixture_timestamp"] = (
        dataframe["date"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    dataframe["prediction_date"] = (
        dataframe["date"]
        .dt.date
        .astype(str)
    )

    numeric_columns = [
        "event_id",
        "round_number",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "goal_difference",
        "total_goals",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Replay input contains missing required "
            "numeric values."
        )

    if (
        dataframe["home_score"].lt(0).any()
        or dataframe["away_score"].lt(0).any()
    ):
        raise ValueError(
            "Replay input contains negative scores."
        )

    expected_goal_difference = (
        dataframe["home_score"]
        - dataframe["away_score"]
    )

    if not np.array_equal(
        dataframe[
            "goal_difference"
        ].to_numpy(),
        expected_goal_difference.to_numpy(),
    ):
        raise ValueError(
            "Replay input contains inconsistent goal "
            "differences."
        )

    expected_total_goals = (
        dataframe["home_score"]
        + dataframe["away_score"]
    )

    if not np.array_equal(
        dataframe[
            "total_goals"
        ].to_numpy(),
        expected_total_goals.to_numpy(),
    ):
        raise ValueError(
            "Replay input contains inconsistent total-goal "
            "values."
        )

    home_clubs = set(
        dataframe["home_team"]
        .astype(str)
        .str.strip()
    )

    away_clubs = set(
        dataframe["away_team"]
        .astype(str)
        .str.strip()
    )

    clubs = home_clubs | away_clubs

    if len(clubs) != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected Bundesliga club population. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {len(clubs)}."
        )

    if dataframe[
        "home_team"
    ].eq(
        dataframe["away_team"]
    ).any():
        raise ValueError(
            "Replay input contains a self-match."
        )

    return (
        dataframe
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def validate_repository_population(
    fixtures: pd.DataFrame,
    repository_clubs: tuple[str, ...],
) -> None:
    """
    Confirm that the historical fixture population and
    production repository contain the same clubs.
    """

    fixture_clubs = set(
        fixtures["home_team"]
        .astype(str)
        .str.strip()
    ) | set(
        fixtures["away_team"]
        .astype(str)
        .str.strip()
    )

    repository_set = set(
        repository_clubs
    )

    missing_from_repository = sorted(
        fixture_clubs - repository_set,
        key=str.casefold,
    )

    extra_in_repository = sorted(
        repository_set - fixture_clubs,
        key=str.casefold,
    )

    if missing_from_repository:
        raise ValueError(
            "Historical fixtures contain clubs missing "
            "from the production repository: "
            f"{missing_from_repository}"
        )

    if extra_in_repository:
        raise ValueError(
            "Production repository contains clubs absent "
            "from the historical fixture population: "
            f"{extra_in_repository}"
        )


def build_replay_summary(
    *,
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    processed_count = (
        len(predictions)
        + len(failures)
    )

    successful_count = len(
        predictions
    )

    failure_count = len(
        failures
    )

    failure_rate = (
        failure_count / processed_count
        if processed_count
        else 0.0
    )

    summary = {
        "input_fixture_count": len(fixtures),
        "processed_fixture_count": processed_count,
        "successful_prediction_count": successful_count,
        "runtime_failure_count": failure_count,
        "runtime_failure_rate": failure_rate,
        "unique_input_event_count": (
            fixtures["event_id"].nunique()
        ),
        "unique_predicted_event_count": (
            predictions["event_id"].nunique()
            if not predictions.empty
            else 0
        ),
        "input_club_count": len(
            set(fixtures["home_team"])
            | set(fixtures["away_team"])
        ),
        "prediction_status": (
            "PASS"
            if failure_count == 0
            else "PARTIAL"
        ),
    }

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value in summary.items()
        ]
    )


def validate_replay_outputs(
    *,
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    if (
        len(predictions)
        + len(failures)
        != len(fixtures)
    ):
        raise AssertionError(
            "Replay output does not account for every "
            "input fixture."
        )

    if not failures.empty:
        return

    if len(predictions) != EXPECTED_MATCH_COUNT:
        raise AssertionError(
            "Successful replay did not preserve all "
            f"{EXPECTED_MATCH_COUNT} fixtures."
        )

    if predictions[
        "event_id"
    ].duplicated().any():
        raise AssertionError(
            "Replay predictions contain duplicate event IDs."
        )

    if set(
        predictions["event_id"]
    ) != set(
        fixtures["event_id"]
    ):
        raise AssertionError(
            "Replay predictions do not preserve the exact "
            "historical event population."
        )

    if not predictions[
        "prediction_status"
    ].eq(
        "PASS"
    ).all():
        raise AssertionError(
            "Successful replay output contains a non-PASS "
            "prediction status."
        )

    numeric_prediction_columns = [
        "home_attack",
        "away_attack",
        "home_defense",
        "away_defense",
        "home_attack_depth",
        "away_attack_depth",
        "attack_depth_diff",
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    values = predictions[
        numeric_prediction_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise AssertionError(
            "Replay predictions contain non-finite values."
        )

    if (
        predictions["lambda_home"].le(0).any()
        or predictions["lambda_away"].le(0).any()
    ):
        raise AssertionError(
            "Replay predictions contain non-positive "
            "expected goals."
        )

    probability_sums = predictions[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].sum(
        axis=1
    )

    if not np.allclose(
        probability_sums.to_numpy(dtype=float),
        np.ones(len(predictions)),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Replay outcome probabilities do not sum to one."
        )


def build_metadata(
    *,
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    failures: pd.DataFrame,
    pipeline,
    repository_path: Path,
    artifact_path: Path,
    study_id: str,
    study_name: str,
    candidate_name: str,
    output_filenames: tuple[str, ...],
) -> dict[str, object]:
    model = pipeline.goal_model

    return {
        "study_id": study_id,
        "study_name": study_name,
        "candidate_name": candidate_name,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "PASS"
            if failures.empty
            else "PARTIAL"
        ),
        "study_type": (
            "operational_production_replay"
        ),
        "accuracy_evaluation_performed": False,
        "interpretation": (
            "This study validates runtime execution over a "
            "complete historical season. It does not provide "
            "an unbiased estimate of predictive performance."
        ),
        "input_dataset": str(
            INPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "competition_key": (
            EXPECTED_COMPETITION_KEY
        ),
        "season_start_year": (
            EXPECTED_SEASON_START_YEAR
        ),
        "fixture_count": len(fixtures),
        "successful_prediction_count": len(
            predictions
        ),
        "runtime_failure_count": len(
            failures
        ),
        "prediction_date_policy": (
            "historical_fixture_calendar_date"
        ),
        "club_repository_path": (
            str(
                repository_path.relative_to(
                    PROJECT_ROOT
                )
            )
            if repository_path.is_relative_to(
                PROJECT_ROOT
            )
            else str(repository_path)
        ),
        "clubelo_cache_directory": str(
            CLUBELO_CACHE_DIRECTORY.relative_to(
                PROJECT_ROOT
            )
        ),
        "goal_model_artifact_path": (
            str(
                artifact_path.relative_to(
                    PROJECT_ROOT
                )
            )
            if artifact_path.is_relative_to(
                PROJECT_ROOT
            )
            else str(artifact_path)
        ),
        "goal_model_artifact_name": (
            model.artifact_name
        ),
        "goal_model_artifact_version": (
            model.artifact_version
        ),
        "goal_model_baseline_name": (
            model.baseline_name
        ),
        "goal_model_baseline_version": (
            model.baseline_version
        ),
        "goal_model_feature_specification": (
            model.feature_specification
        ),
        "goal_model_training_end_date": (
            model.training_end_date
        ),
        "production_repository_scope": (
            "bundesliga_2024_25"
        ),
        "outputs": list(
            output_filenames
        ),
    }


def main() -> None:
    arguments = parse_arguments()

    repository_path = (
        arguments.repository_path
    )

    artifact_path = (
        arguments.artifact_path
    )

    output_directory = (
        arguments.output_directory
    )

    replay_predictions_path = (
        output_directory
        / "fixture_replay_predictions.csv"
    )

    replay_summary_path = (
        output_directory
        / "replay_summary.csv"
    )

    runtime_failures_path = (
        output_directory
        / "runtime_failures.csv"
    )

    metadata_path = (
        output_directory
        / "study_metadata.json"
    )

    print("=" * 88)
    print(
        f"{arguments.study_id} — "
        f"{arguments.study_name}"
    )
    print("=" * 88)

    fixtures = load_replay_population()

    pipeline = build_production_prediction_pipeline(
        club_repository_path=(
            repository_path
        ),
        clubelo_cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        ),
        goal_model_artifact_path=(
            artifact_path
        ),
        clubelo_name_overrides=(
            BUNDESLIGA_CLUBELO_NAME_OVERRIDES
        ),
    )

    repository_clubs = (
        pipeline
        .observation_builder
        .club_repository
        .list_clubs()
    )

    validate_repository_population(
        fixtures,
        repository_clubs,
    )

    print()
    print("Replay configuration")
    print(
        f"  Candidate: "
        f"{arguments.candidate_name}"
    )
    print(
        f"  Repository: {repository_path}"
    )
    print(
        f"  Goal artifact: {artifact_path}"
    )
    print(
        f"  Input fixtures: {len(fixtures)}"
    )
    print(
        "  Fixture period: "
        f"{fixtures['prediction_date'].min()} "
        "through "
        f"{fixtures['prediction_date'].max()}"
    )
    print(
        f"  Clubs: {len(repository_clubs)}"
    )
    print(
        "  Goal model: "
        f"{pipeline.goal_model.artifact_name} "
        f"v{pipeline.goal_model.artifact_version}"
    )
    print(
        "  Model training cutoff: "
        f"{pipeline.goal_model.training_end_date}"
    )
    print(
        "  Evaluation mode: operational replay only"
    )

    print()
    print(
        "Replaying all completed fixtures..."
    )

    predictions, failures = (
        pipeline.predict_fixtures(
            fixtures,
            prediction_date_column=(
                "prediction_date"
            ),
            continue_on_error=True,
        )
    )

    validate_replay_outputs(
        fixtures=fixtures,
        predictions=predictions,
        failures=failures,
    )

    summary = build_replay_summary(
        fixtures=fixtures,
        predictions=predictions,
        failures=failures,
    )

    metadata = build_metadata(
        fixtures=fixtures,
        predictions=predictions,
        failures=failures,
        pipeline=pipeline,
        repository_path=repository_path,
        artifact_path=artifact_path,
        study_id=arguments.study_id,
        study_name=arguments.study_name,
        candidate_name=(
            arguments.candidate_name
        ),
        output_filenames=(
            replay_predictions_path.name,
            replay_summary_path.name,
            runtime_failures_path.name,
            metadata_path.name,
        ),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        replay_predictions_path,
        index=False,
    )

    summary.to_csv(
        replay_summary_path,
        index=False,
    )

    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "event_id",
                "fixture_row_index",
                "requested_home_team",
                "requested_away_team",
                "requested_prediction_date",
                "prediction_status",
                "runtime_error_type",
                "runtime_error",
            ]
        )

    failures.to_csv(
        runtime_failures_path,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Replay summary")
    print("-" * 88)
    print(
        f"  Fixtures loaded: {len(fixtures)}"
    )
    print(
        "  Successful predictions: "
        f"{len(predictions)}"
    )
    print(
        f"  Runtime failures: {len(failures)}"
    )
    print(
        "  Event population preserved: "
        f"{predictions['event_id'].nunique()}"
        f"/{fixtures['event_id'].nunique()}"
    )

    print()
    print("Validation summary")
    print("  Canonical input loading: PASS")
    print("  Completed-match validation: PASS")
    print("  Club population alignment: PASS")
    print("  Historical date parsing: PASS")
    print(
        "  Full fixture accounting: "
        + (
            "PASS"
            if (
                len(predictions)
                + len(failures)
                == len(fixtures)
            )
            else "FAIL"
        )
    )
    print(
        "  Observation construction: "
        + (
            "PASS"
            if failures.empty
            else "PARTIAL"
        )
    )
    print(
        "  Goal prediction generation: "
        + (
            "PASS"
            if failures.empty
            else "PARTIAL"
        )
    )
    print(
        "  Outcome probability generation: "
        + (
            "PASS"
            if failures.empty
            else "PARTIAL"
        )
    )
    print(
        "  Runtime failure count: "
        f"{len(failures)}"
    )
    print("  Accuracy evaluation excluded: PASS")

    print()

    if failures.empty:
        print("=" * 88)
        print("OVERALL RESULT: PASS")
        print("=" * 88)
    else:
        print("Runtime failures")
        print("-" * 88)
        print(
            failures.to_string(
                index=False
            )
        )
        print()
        print("=" * 88)
        print("OVERALL RESULT: PARTIAL")
        print("=" * 88)

        raise SystemExit(1)

    print()
    print(
        f"Outputs written to: "
        f"{output_directory}"
    )


if __name__ == "__main__":
    main()