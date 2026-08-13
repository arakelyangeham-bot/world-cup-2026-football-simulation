#generate_prequential_goal_expectations

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.baselines.club_goal_model import (
    CURRENT_CLUB_GOAL_MODEL,
)
from simulation.goal_models import (
    GoalPrediction,
    PoissonGoalModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
    / "full_squad_observations_with_clubelo.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_064_prequential_goal_expectations"
)

EXPECTATION_PATH = (
    OUTPUT_DIRECTORY
    / "historical_goal_expectations.csv"
)

AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "prediction_generation_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


MINIMUM_TRAINING_MATCHES = 80

MODEL_ALPHA = 0.0

REQUIRED_IDENTITY_COLUMNS = (
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
)

REQUIRED_TARGET_COLUMNS = (
    "home_score",
    "away_score",
)

OUTPUT_COLUMNS = (
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
    "actual_home_goals",
    "actual_away_goals",
    "expected_home_goals",
    "expected_away_goals",
    "prediction_available",
    "prediction_temporal_validity_pass",
    "training_match_count",
    "training_start_date",
    "training_end_date",
    "prediction_date",
    "baseline_name",
    "baseline_version",
    "feature_specification",
    "model_alpha",
    "unavailability_reason",
)


def load_observations() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Study 060 enriched observations do not "
            f"exist: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 060 enriched observations are empty."
        )

    specification = (
        CURRENT_CLUB_GOAL_MODEL
        .get_feature_specification()
    )

    required_columns = {
        *REQUIRED_IDENTITY_COLUMNS,
        *REQUIRED_TARGET_COLUMNS,
        *specification.required_columns(),
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Enriched observations are missing required "
            f"columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        *REQUIRED_TARGET_COLUMNS,
        *specification.required_columns(),
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
            "Required target or feature columns contain "
            "missing numeric values."
        )

    if not np.isfinite(
        dataframe[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Required target or feature columns contain "
            "non-finite values."
        )

    if dataframe[
        "event_id"
    ].isna().any():
        raise ValueError(
            "Observation dataset contains missing "
            "event IDs."
        )

    if dataframe[
        "event_id"
    ].duplicated().any():
        raise ValueError(
            "Observation dataset contains duplicate "
            "event IDs."
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


def build_goal_model(
    training_match_count: int,
    prediction_date: pd.Timestamp,
) -> PoissonGoalModel:
    baseline = CURRENT_CLUB_GOAL_MODEL
    specification = (
        baseline.get_feature_specification()
    )

    model_name = (
        f"{baseline.name}_"
        f"v{baseline.version}_"
        f"train_{training_match_count}_"
        f"predict_{prediction_date.date().isoformat()}"
    )

    return PoissonGoalModel(
        name=model_name,
        home_features=list(
            specification.home_features
        ),
        away_features=list(
            specification.away_features
        ),
        alpha=MODEL_ALPHA,
    )


def unavailable_rows(
    prediction_batch: pd.DataFrame,
    training_match_count: int,
    reason: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "event_id":
                prediction_batch["event_id"],
            "date":
                prediction_batch["date"],
            "home_team":
                prediction_batch["home_team"],
            "home_team_id":
                prediction_batch["home_team_id"],
            "away_team":
                prediction_batch["away_team"],
            "away_team_id":
                prediction_batch["away_team_id"],
            "actual_home_goals":
                prediction_batch["home_score"],
            "actual_away_goals":
                prediction_batch["away_score"],
            "expected_home_goals":
                np.nan,
            "expected_away_goals":
                np.nan,
            "prediction_available":
                False,
            "prediction_temporal_validity_pass":
                False,
            "training_match_count":
                training_match_count,
            "training_start_date":
                pd.NaT,
            "training_end_date":
                pd.NaT,
            "prediction_date":
                prediction_batch["date"],
            "baseline_name":
                CURRENT_CLUB_GOAL_MODEL.name,
            "baseline_version":
                CURRENT_CLUB_GOAL_MODEL.version,
            "feature_specification":
                CURRENT_CLUB_GOAL_MODEL
                .feature_specification,
            "model_alpha":
                MODEL_ALPHA,
            "unavailability_reason":
                reason,
        }
    )

    return output[
        list(OUTPUT_COLUMNS)
    ]


def available_rows(
    prediction_batch: pd.DataFrame,
    training_data: pd.DataFrame,
    prediction: GoalPrediction,
) -> pd.DataFrame:
    prediction_date = (
        prediction_batch["date"].iloc[0]
    )

    training_start_date = (
        training_data["date"].min()
    )

    training_end_date = (
        training_data["date"].max()
    )

    temporal_validity_pass = bool(
        training_end_date.date()
        < prediction_date.date()
    )

    if not temporal_validity_pass:
        raise AssertionError(
            "Prequential training data are not strictly "
            "earlier than the prediction date."
        )

    expected_home = np.asarray(
        prediction.pred_home_goals,
        dtype=float,
    )

    expected_away = np.asarray(
        prediction.pred_away_goals,
        dtype=float,
    )

    if len(expected_home) != len(
        prediction_batch
    ):
        raise AssertionError(
            "Home prediction count does not match the "
            "prediction batch."
        )

    if len(expected_away) != len(
        prediction_batch
    ):
        raise AssertionError(
            "Away prediction count does not match the "
            "prediction batch."
        )

    if not np.isfinite(
        expected_home
    ).all():
        raise AssertionError(
            "Home predictions contain non-finite values."
        )

    if not np.isfinite(
        expected_away
    ).all():
        raise AssertionError(
            "Away predictions contain non-finite values."
        )

    if (
        (expected_home <= 0.0).any()
        or (expected_away <= 0.0).any()
    ):
        raise AssertionError(
            "Poisson expectations must be positive."
        )

    output = pd.DataFrame(
        {
            "event_id":
                prediction_batch["event_id"]
                .to_numpy(),
            "date":
                prediction_batch["date"]
                .to_numpy(),
            "home_team":
                prediction_batch["home_team"]
                .to_numpy(),
            "home_team_id":
                prediction_batch["home_team_id"]
                .to_numpy(),
            "away_team":
                prediction_batch["away_team"]
                .to_numpy(),
            "away_team_id":
                prediction_batch["away_team_id"]
                .to_numpy(),
            "actual_home_goals":
                prediction_batch["home_score"]
                .to_numpy(),
            "actual_away_goals":
                prediction_batch["away_score"]
                .to_numpy(),
            "expected_home_goals":
                expected_home,
            "expected_away_goals":
                expected_away,
            "prediction_available":
                True,
            "prediction_temporal_validity_pass":
                temporal_validity_pass,
            "training_match_count":
                len(training_data),
            "training_start_date":
                training_start_date,
            "training_end_date":
                training_end_date,
            "prediction_date":
                prediction_batch["date"]
                .to_numpy(),
            "baseline_name":
                CURRENT_CLUB_GOAL_MODEL.name,
            "baseline_version":
                CURRENT_CLUB_GOAL_MODEL.version,
            "feature_specification":
                CURRENT_CLUB_GOAL_MODEL
                .feature_specification,
            "model_alpha":
                MODEL_ALPHA,
            "unavailability_reason":
                "",
        }
    )

    return output[
        list(OUTPUT_COLUMNS)
    ]


def generate_prequential_expectations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate date-batched expanding-window predictions.

    All matches on one calendar date are predicted from a
    model trained only on matches from strictly earlier
    calendar dates.
    """
    output_frames: list[pd.DataFrame] = []

    calendar_dates = (
        observations["date"]
        .dt.date
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for calendar_date in calendar_dates:
        prediction_mask = (
            observations["date"].dt.date
            == calendar_date
        )

        training_mask = (
            observations["date"].dt.date
            < calendar_date
        )

        prediction_batch = (
            observations.loc[
                prediction_mask
            ]
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .reset_index(drop=True)
        )

        training_data = (
            observations.loc[
                training_mask
            ]
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .reset_index(drop=True)
        )

        training_match_count = len(
            training_data
        )

        if (
            training_match_count
            < MINIMUM_TRAINING_MATCHES
        ):
            output_frames.append(
                unavailable_rows(
                    prediction_batch=(
                        prediction_batch
                    ),
                    training_match_count=(
                        training_match_count
                    ),
                    reason=(
                        "minimum_training_matches_"
                        "not_reached"
                    ),
                )
            )

            continue

        model = build_goal_model(
            training_match_count=(
                training_match_count
            ),
            prediction_date=(
                prediction_batch[
                    "date"
                ].iloc[0]
            ),
        )

        model.fit(training_data)

        prediction = model.predict(
            prediction_batch
        )

        output_frames.append(
            available_rows(
                prediction_batch=(
                    prediction_batch
                ),
                training_data=training_data,
                prediction=prediction,
            )
        )

    if not output_frames:
        raise AssertionError(
            "No prequential expectation rows were "
            "generated."
        )

    output = pd.concat(
        output_frames,
        ignore_index=True,
    )

    return (
        output
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def validate_row_preservation(
    observations: pd.DataFrame,
    expectations: pd.DataFrame,
) -> None:
    if len(expectations) != len(
        observations
    ):
        raise AssertionError(
            "Expectation generation changed the match "
            f"population: {len(expectations)} vs "
            f"{len(observations)}."
        )

    if expectations[
        "event_id"
    ].duplicated().any():
        raise AssertionError(
            "Expectation output contains duplicate "
            "event IDs."
        )

    source_ids = set(
        observations["event_id"]
    )

    output_ids = set(
        expectations["event_id"]
    )

    if source_ids != output_ids:
        missing = sorted(
            source_ids - output_ids
        )

        unexpected = sorted(
            output_ids - source_ids
        )

        raise AssertionError(
            "Expectation event population differs from "
            "the source population. "
            f"Missing: {missing[:20]}; "
            f"unexpected: {unexpected[:20]}."
        )


def validate_population_identity(
    observations: pd.DataFrame,
    expectations: pd.DataFrame,
) -> None:
    source = (
        observations[
            [
                "event_id",
                "date",
                "home_team",
                "home_team_id",
                "away_team",
                "away_team_id",
                "home_score",
                "away_score",
            ]
        ]
        .set_index("event_id")
        .sort_index()
    )

    output = (
        expectations
        .set_index("event_id")
        .sort_index()
    )

    if not source.index.equals(
        output.index
    ):
        raise AssertionError(
            "Source and expectation event indices differ."
        )

    comparisons = (
        (
            "date",
            "date",
        ),
        (
            "home_team",
            "home_team",
        ),
        (
            "home_team_id",
            "home_team_id",
        ),
        (
            "away_team",
            "away_team",
        ),
        (
            "away_team_id",
            "away_team_id",
        ),
        (
            "home_score",
            "actual_home_goals",
        ),
        (
            "away_score",
            "actual_away_goals",
        ),
    )

    for source_column, output_column in (
        comparisons
    ):
        left = source[source_column]
        right = output[output_column]

        if pd.api.types.is_numeric_dtype(
            left
        ):
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
            ).all()
        else:
            equal = (
                left.fillna("<missing>")
                .astype(str)
                .eq(
                    right.fillna("<missing>")
                    .astype(str)
                )
                .all()
            )

        if not equal:
            raise AssertionError(
                "Expectation output changed source "
                f"values for {source_column!r}."
            )


def validate_availability_contract(
    expectations: pd.DataFrame,
) -> None:
    available = expectations[
        "prediction_available"
    ].astype(bool)

    unavailable = ~available

    if not available.any():
        raise AssertionError(
            "No prequential predictions became "
            "available."
        )

    if not unavailable.any():
        raise AssertionError(
            "Warm-up rows were unexpectedly absent."
        )

    if expectations.loc[
        available,
        [
            "expected_home_goals",
            "expected_away_goals",
        ],
    ].isna().any().any():
        raise AssertionError(
            "Available predictions contain missing "
            "expected goals."
        )

    if expectations.loc[
        unavailable,
        [
            "expected_home_goals",
            "expected_away_goals",
        ],
    ].notna().any().any():
        raise AssertionError(
            "Unavailable warm-up rows contain expected "
            "goals."
        )

    if not expectations.loc[
        available,
        "unavailability_reason",
    ].fillna("").eq("").all():
        raise AssertionError(
            "Available predictions contain an "
            "unavailability reason."
        )

    expected_reason = (
        "minimum_training_matches_not_reached"
    )

    if not expectations.loc[
        unavailable,
        "unavailability_reason",
    ].eq(expected_reason).all():
        raise AssertionError(
            "Warm-up rows contain an unexpected "
            "unavailability reason."
        )

    if not (
        expectations.loc[
            unavailable,
            "training_match_count",
        ]
        < MINIMUM_TRAINING_MATCHES
    ).all():
        raise AssertionError(
            "Unavailable rows do not satisfy the "
            "warm-up threshold rule."
        )

    if not (
        expectations.loc[
            available,
            "training_match_count",
        ]
        >= MINIMUM_TRAINING_MATCHES
    ).all():
        raise AssertionError(
            "Available rows do not satisfy the "
            "minimum training threshold."
        )


def validate_temporal_integrity(
    expectations: pd.DataFrame,
) -> None:
    available = expectations[
        expectations[
            "prediction_available"
        ]
    ].copy()

    available[
        "training_end_date"
    ] = pd.to_datetime(
        available[
            "training_end_date"
        ],
        errors="raise",
        utc=True,
    )

    available[
        "prediction_date"
    ] = pd.to_datetime(
        available[
            "prediction_date"
        ],
        errors="raise",
        utc=True,
    )

    strictly_prior = (
        available[
            "training_end_date"
        ].dt.date
        < available[
            "prediction_date"
        ].dt.date
    )

    if not strictly_prior.all():
        raise AssertionError(
            "One or more prequential predictions used "
            "same-day or future training data."
        )

    if not available[
        "prediction_temporal_validity_pass"
    ].all():
        raise AssertionError(
            "One or more available predictions failed "
            "the temporal-validity flag."
        )

    unavailable = expectations[
        ~expectations[
            "prediction_available"
        ]
    ]

    if unavailable[
        "prediction_temporal_validity_pass"
    ].any():
        raise AssertionError(
            "Unavailable rows were incorrectly marked "
            "temporally valid."
        )


def validate_date_batching(
    expectations: pd.DataFrame,
) -> None:
    available = expectations[
        expectations[
            "prediction_available"
        ]
    ].copy()

    grouped = available.groupby(
        available["date"].dt.date,
        sort=True,
    )

    for calendar_date, group in grouped:
        if group[
            "training_match_count"
        ].nunique() != 1:
            raise AssertionError(
                "Matches sharing one calendar date used "
                "different training populations: "
                f"{calendar_date}"
            )

        if group[
            "training_end_date"
        ].nunique() != 1:
            raise AssertionError(
                "Matches sharing one calendar date used "
                "different training cutoffs: "
                f"{calendar_date}"
            )


def validate_training_progression(
    expectations: pd.DataFrame,
) -> None:
    date_summary = (
        expectations
        .assign(
            calendar_date=(
                expectations["date"].dt.date
            )
        )
        .groupby(
            "calendar_date",
            as_index=False,
        )
        .agg(
            training_match_count=(
                "training_match_count",
                "first",
            ),
            prediction_batch_size=(
                "event_id",
                "size",
            ),
        )
        .sort_values("calendar_date")
        .reset_index(drop=True)
    )

    counts = date_summary[
        "training_match_count"
    ].to_numpy(dtype=int)

    if np.any(
        np.diff(counts) < 0
    ):
        raise AssertionError(
            "Training-match counts decreased over time."
        )

    expected_next_counts = (
        date_summary[
            "training_match_count"
        ].iloc[:-1]
        .to_numpy(dtype=int)
        + date_summary[
            "prediction_batch_size"
        ].iloc[:-1]
        .to_numpy(dtype=int)
    )

    observed_next_counts = (
        date_summary[
            "training_match_count"
        ].iloc[1:]
        .to_numpy(dtype=int)
    )

    if not np.array_equal(
        expected_next_counts,
        observed_next_counts,
    ):
        raise AssertionError(
            "Training populations do not advance by the "
            "complete preceding date batch."
        )


def validate_baseline_contract(
    expectations: pd.DataFrame,
) -> None:
    baseline = CURRENT_CLUB_GOAL_MODEL

    if not expectations[
        "baseline_name"
    ].eq(baseline.name).all():
        raise AssertionError(
            "Unexpected baseline name in expectation "
            "output."
        )

    if not expectations[
        "baseline_version"
    ].astype(str).eq(
        baseline.version
    ).all():
        raise AssertionError(
            "Unexpected baseline version in expectation "
            "output."
        )

    if not expectations[
        "feature_specification"
    ].eq(
        baseline.feature_specification
    ).all():
        raise AssertionError(
            "Unexpected feature specification in "
            "expectation output."
        )

    if not np.isclose(
        expectations[
            "model_alpha"
        ].to_numpy(dtype=float),
        MODEL_ALPHA,
        atol=0.0,
        rtol=0.0,
    ).all():
        raise AssertionError(
            "Unexpected alpha value in expectation "
            "output."
        )


def build_audit(
    expectations: pd.DataFrame,
) -> pd.DataFrame:
    audit = expectations[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "prediction_available",
            "prediction_temporal_validity_pass",
            "training_match_count",
            "training_start_date",
            "training_end_date",
            "prediction_date",
            "unavailability_reason",
        ]
    ].copy()

    audit[
        "warmup_contract_pass"
    ] = np.where(
        audit["prediction_available"],
        audit[
            "training_match_count"
        ] >= MINIMUM_TRAINING_MATCHES,
        audit[
            "training_match_count"
        ] < MINIMUM_TRAINING_MATCHES,
    )

    audit[
        "expected_goal_contract_pass"
    ] = np.where(
        audit["prediction_available"],
        expectations[
            [
                "expected_home_goals",
                "expected_away_goals",
            ]
        ].notna().all(axis=1),
        expectations[
            [
                "expected_home_goals",
                "expected_away_goals",
            ]
        ].isna().all(axis=1),
    )

    audit[
        "overall_validation_pass"
    ] = (
        audit["warmup_contract_pass"]
        & audit[
            "expected_goal_contract_pass"
        ]
        & np.where(
            audit["prediction_available"],
            audit[
                "prediction_temporal_validity_pass"
            ],
            True,
        )
    )

    return audit


def build_metadata(
    observations: pd.DataFrame,
    expectations: pd.DataFrame,
) -> dict[str, object]:
    available = expectations[
        expectations[
            "prediction_available"
        ]
    ]

    unavailable = expectations[
        ~expectations[
            "prediction_available"
        ]
    ]

    first_available_date = (
        available["date"].min()
        if not available.empty
        else None
    )

    return {
        "study_id": "064",
        "study_name": (
            "Leakage-Safe Prequential Goal "
            "Expectations"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_path": str(INPUT_PATH),
        "output_path": str(
            EXPECTATION_PATH
        ),
        "baseline_name":
            CURRENT_CLUB_GOAL_MODEL.name,
        "baseline_version":
            CURRENT_CLUB_GOAL_MODEL.version,
        "feature_specification":
            CURRENT_CLUB_GOAL_MODEL
            .feature_specification,
        "model_alpha": MODEL_ALPHA,
        "minimum_training_matches":
            MINIMUM_TRAINING_MATCHES,
        "date_batching_enabled": True,
        "strictly_prior_dates_only": True,
        "source_match_count": int(
            len(observations)
        ),
        "output_match_count": int(
            len(expectations)
        ),
        "available_prediction_count": int(
            len(available)
        ),
        "unavailable_warmup_count": int(
            len(unavailable)
        ),
        "prediction_coverage_rate": float(
            len(available)
            / len(expectations)
        ),
        "unique_prediction_dates": int(
            expectations[
                "date"
            ].dt.date.nunique()
        ),
        "first_available_prediction_date": (
            first_available_date.isoformat()
            if first_available_date is not None
            else None
        ),
        "mean_expected_home_goals": (
            float(
                available[
                    "expected_home_goals"
                ].mean()
            )
            if not available.empty
            else None
        ),
        "mean_expected_away_goals": (
            float(
                available[
                    "expected_away_goals"
                ].mean()
            )
            if not available.empty
            else None
        ),
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 064 — Leakage-Safe Prequential Goal Expectations

## Purpose

Generate historically valid Version 1 goal expectations for
the club observation population.

The study uses date-batched expanding-window prediction:

1. Train on matches from calendar dates strictly before the
   prediction date.
2. Predict every match on the prediction date.
3. Advance the training pool only after the complete date
   batch has been predicted.

## Baseline

- Name: `{metadata["baseline_name"]}`
- Version: `{metadata["baseline_version"]}`
- Feature specification:
  `{metadata["feature_specification"]}`
- Alpha: `{metadata["model_alpha"]}`

## Warm-up policy

- Minimum training matches:
  {metadata["minimum_training_matches"]}
- Early matches remain in the output population.
- Warm-up matches have `prediction_available = False`.
- No expected-goal values are fabricated during warm-up.

## Population

- Source matches: {metadata["source_match_count"]}
- Output matches: {metadata["output_match_count"]}
- Available predictions:
  {metadata["available_prediction_count"]}
- Unavailable warm-up rows:
  {metadata["unavailable_warmup_count"]}
- Prediction coverage:
  {metadata["prediction_coverage_rate"]:.2%}
- Unique prediction dates:
  {metadata["unique_prediction_dates"]}
- First available prediction date:
  {metadata["first_available_prediction_date"]}

## Expected goals

- Mean expected home goals:
  {metadata["mean_expected_home_goals"]:.6f}
- Mean expected away goals:
  {metadata["mean_expected_away_goals"]:.6f}

## Validation

- Source row preservation: PASS
- Event-population identity: PASS
- Duplicate-event prevention: PASS
- Warm-up availability contract: PASS
- Strict prior-date training: PASS
- Same-date batch consistency: PASS
- Training-population progression: PASS
- Baseline registry integration: PASS
- Baseline version consistency: PASS
- Feature-specification consistency: PASS
- Positive finite expected goals: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    CURRENT_CLUB_GOAL_MODEL.validate()

    observations = load_observations()

    expectations = (
        generate_prequential_expectations(
            observations
        )
    )

    validate_row_preservation(
        observations=observations,
        expectations=expectations,
    )

    validate_population_identity(
        observations=observations,
        expectations=expectations,
    )

    validate_availability_contract(
        expectations
    )

    validate_temporal_integrity(
        expectations
    )

    validate_date_batching(
        expectations
    )

    validate_training_progression(
        expectations
    )

    validate_baseline_contract(
        expectations
    )

    audit = build_audit(
        expectations
    )

    if not audit[
        "overall_validation_pass"
    ].all():
        failures = audit[
            ~audit[
                "overall_validation_pass"
            ]
        ]

        print(
            failures.head(30).to_string(
                index=False
            )
        )

        raise AssertionError(
            "One or more prediction-generation audit "
            "rows failed."
        )

    metadata = build_metadata(
        observations=observations,
        expectations=expectations,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = expectations.copy()

    date_columns = (
        "date",
        "training_start_date",
        "training_end_date",
        "prediction_date",
    )

    for column in date_columns:
        output[column] = pd.to_datetime(
            output[column],
            errors="coerce",
            utc=True,
        ).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        output[column] = output[
            column
        ].replace(
            {
                "NaT": "",
                "nan": "",
            }
        )

    output.to_csv(
        EXPECTATION_PATH,
        index=False,
    )

    audit_output = audit.copy()

    for column in (
        "date",
        "training_start_date",
        "training_end_date",
        "prediction_date",
    ):
        audit_output[column] = pd.to_datetime(
            audit_output[column],
            errors="coerce",
            utc=True,
        ).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        audit_output[column] = (
            audit_output[column]
            .replace(
                {
                    "NaT": "",
                    "nan": "",
                }
            )
        )

    audit_output.to_csv(
        AUDIT_PATH,
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

    available = expectations[
        expectations[
            "prediction_available"
        ]
    ]

    unavailable = expectations[
        ~expectations[
            "prediction_available"
        ]
    ]

    print(
        "Study 064 — Leakage-Safe "
        "Prequential Goal Expectations"
    )
    print("=" * 76)
    print()
    print(
        f"Baseline: "
        f"{CURRENT_CLUB_GOAL_MODEL.name}"
    )
    print(
        f"Version: "
        f"{CURRENT_CLUB_GOAL_MODEL.version}"
    )
    print(
        "Feature specification: "
        f"{CURRENT_CLUB_GOAL_MODEL.feature_specification}"
    )
    print(
        f"Model alpha: {MODEL_ALPHA}"
    )
    print(
        "Minimum training matches: "
        f"{MINIMUM_TRAINING_MATCHES}"
    )
    print()
    print(
        f"Source matches: {len(observations)}"
    )
    print(
        f"Output rows: {len(expectations)}"
    )
    print(
        "Available predictions: "
        f"{len(available)}"
    )
    print(
        "Unavailable warm-up rows: "
        f"{len(unavailable)}"
    )
    print(
        "Prediction coverage: "
        f"{len(available) / len(expectations):.2%}"
    )

    if not available.empty:
        print(
            "First available prediction date: "
            f"{available['date'].min()}"
        )
        print(
            "Mean expected home goals: "
            f"{available['expected_home_goals'].mean():.6f}"
        )
        print(
            "Mean expected away goals: "
            f"{available['expected_away_goals'].mean():.6f}"
        )

    print()
    print("Source row preservation: PASS")
    print("Event-population identity: PASS")
    print("Warm-up availability contract: PASS")
    print("Strict prior-date training: PASS")
    print("Same-date batching: PASS")
    print("Training-population progression: PASS")
    print("Baseline registry integration: PASS")
    print("Expected-goal validation: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()