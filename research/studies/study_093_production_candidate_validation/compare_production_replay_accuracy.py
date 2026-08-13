#compare_production_replay_accuracy

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GLOBAL_REPLAY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_093_production_candidate_validation"
    / "paired_replays"
    / "global_zscore"
    / "fixture_replay_predictions.csv"
)

ROBUST_REPLAY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_093_production_candidate_validation"
    / "paired_replays"
    / "robust_zscore"
    / "fixture_replay_predictions.csv"
)

SPLIT_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c2"
    / "representation_benchmark"
    / "split_assignments.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_093_production_candidate_validation"
    / "replay_accuracy_comparison"
)

PERFORMANCE_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_performance.csv"
)

PAIRED_FIXTURE_PATH = (
    OUTPUT_DIRECTORY
    / "paired_fixture_comparison.csv"
)

METRIC_DELTA_PATH = (
    OUTPUT_DIRECTORY
    / "metric_deltas.csv"
)

LARGEST_CHANGES_PATH = (
    OUTPUT_DIRECTORY
    / "largest_prediction_changes.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_093c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_093C_RESULTS.md"
)


EXPECTED_HOLDOUT_COUNT = 77
TRAIN_FRACTION = 0.75
PROBABILITY_FLOOR = 1e-15


PRIMARY_METRICS = (
    "combined_poisson_deviance",
    "combined_goal_mae",
    "total_goal_mae",
    "goal_difference_mae",
    "outcome_log_loss",
    "outcome_brier_score",
    "exact_score_log_loss",
)

CALIBRATION_METRICS = (
    "absolute_draw_rate_error",
    "absolute_home_goal_mean_error",
    "absolute_away_goal_mean_error",
)

COMPARISON_METRICS = (
    *PRIMARY_METRICS,
    *CALIBRATION_METRICS,
)


IDENTITY_AND_TARGET_COLUMNS = (
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
    "home_score",
    "away_score",
    "total_goals",
    "goal_difference",
)


REQUIRED_REPLAY_COLUMNS = {
    *IDENTITY_AND_TARGET_COLUMNS,
    "lambda_home",
    "lambda_away",
    "pred_total_goals",
    "pred_goal_diff",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "prediction_status",
    "goal_model_artifact_name",
    "goal_model_artifact_version",
    "goal_model_feature_specification",
    "goal_model_training_end_date",
}


def load_holdout_event_ids() -> set[int]:
    if not SPLIT_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            "Split-assignment file does not exist: "
            f"{SPLIT_ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_csv(
        SPLIT_ASSIGNMENTS_PATH,
        low_memory=False,
    )

    required_columns = {
        "event_id",
        "train_fraction",
        "split",
    }

    missing = (
        required_columns
        - set(assignments.columns)
    )

    if missing:
        raise ValueError(
            "Split assignments are missing columns: "
            f"{sorted(missing)}"
        )

    selected = assignments.loc[
        np.isclose(
            pd.to_numeric(
                assignments["train_fraction"],
                errors="raise",
            ).to_numpy(dtype=float),
            TRAIN_FRACTION,
            atol=1e-12,
            rtol=0.0,
        )
        & assignments["split"].eq("test")
    ].copy()

    if len(selected) != EXPECTED_HOLDOUT_COUNT:
        raise AssertionError(
            "Unexpected holdout population: "
            f"{len(selected)} versus "
            f"{EXPECTED_HOLDOUT_COUNT}."
        )

    if selected["event_id"].duplicated().any():
        raise AssertionError(
            "Holdout split contains duplicate event IDs."
        )

    return set(
        pd.to_numeric(
            selected["event_id"],
            errors="raise",
        ).astype(int)
    )


def load_candidate_replay(
    *,
    path: Path,
    candidate: str,
    holdout_event_ids: set[int],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{candidate}: replay file does not exist: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"{candidate}: replay file is empty."
        )

    missing = (
        REQUIRED_REPLAY_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{candidate}: replay file is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["event_id"] = pd.to_numeric(
        dataframe["event_id"],
        errors="raise",
    ).astype(int)

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            f"{candidate}: replay contains duplicate "
            "event IDs."
        )

    if not dataframe[
        "prediction_status"
    ].eq("PASS").all():
        raise ValueError(
            f"{candidate}: replay contains non-PASS rows."
        )

    holdout = dataframe.loc[
        dataframe["event_id"].isin(
            holdout_event_ids
        )
    ].copy()

    if len(holdout) != EXPECTED_HOLDOUT_COUNT:
        raise AssertionError(
            f"{candidate}: holdout row count is "
            f"{len(holdout)}, expected "
            f"{EXPECTED_HOLDOUT_COUNT}."
        )

    if set(holdout["event_id"]) != holdout_event_ids:
        raise AssertionError(
            f"{candidate}: holdout event population differs "
            "from the frozen split."
        )

    numeric_columns = [
        "home_score",
        "away_score",
        "total_goals",
        "goal_difference",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    for column in numeric_columns:
        holdout[column] = pd.to_numeric(
            holdout[column],
            errors="raise",
        )

    values = holdout[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{candidate}: holdout contains non-finite "
            "numeric values."
        )

    if (
        holdout["lambda_home"].le(0).any()
        or holdout["lambda_away"].le(0).any()
    ):
        raise ValueError(
            f"{candidate}: holdout contains non-positive "
            "expected goals."
        )

    probability_sums = holdout[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].sum(axis=1)

    if not np.allclose(
        probability_sums.to_numpy(dtype=float),
        np.ones(len(holdout)),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError(
            f"{candidate}: outcome probabilities do not "
            "sum to one."
        )

    return (
        holdout
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def validate_matched_candidates(
    global_replay: pd.DataFrame,
    robust_replay: pd.DataFrame,
) -> None:
    global_indexed = (
        global_replay
        .set_index("event_id")
        .sort_index()
    )

    robust_indexed = (
        robust_replay
        .set_index("event_id")
        .sort_index()
    )

    if not global_indexed.index.equals(
        robust_indexed.index
    ):
        raise AssertionError(
            "Candidate holdout event populations differ."
        )

    comparison_columns = [
        column
        for column in IDENTITY_AND_TARGET_COLUMNS
        if column != "event_id"
    ]

    for column in comparison_columns:
        left = global_indexed[column]
        right = robust_indexed[column]

        if pd.api.types.is_numeric_dtype(left):
            equal = np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                atol=0.0,
                rtol=0.0,
            )
        else:
            equal = (
                left.fillna("<missing>")
                .astype(str)
                .equals(
                    right.fillna("<missing>")
                    .astype(str)
                )
            )

        if not equal:
            raise AssertionError(
                "Candidate replays disagree on matched "
                f"column {column!r}."
            )

    global_specifications = set(
        global_replay[
            "goal_model_feature_specification"
        ]
    )

    robust_specifications = set(
        robust_replay[
            "goal_model_feature_specification"
        ]
    )

    if global_specifications != robust_specifications:
        raise AssertionError(
            "Candidate goal-model feature "
            "specifications differ."
        )

    global_cutoffs = set(
        global_replay[
            "goal_model_training_end_date"
        ]
    )

    robust_cutoffs = set(
        robust_replay[
            "goal_model_training_end_date"
        ]
    )

    if global_cutoffs != robust_cutoffs:
        raise AssertionError(
            "Candidate training cutoffs differ."
        )


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    log_probability = (
        -expected_goals
        + goals * math.log(expected_goals)
        - math.lgamma(goals + 1)
    )

    return math.exp(log_probability)


def observed_result_index(
    home_score: int,
    away_score: int,
) -> int:
    if home_score > away_score:
        return 0

    if home_score == away_score:
        return 1

    return 2


def multiclass_brier_score(
    observed_indices: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(len(observed_indices)),
        observed_indices,
    ] = 1.0

    return float(
        (
            (
                probabilities
                - one_hot
            )
            ** 2
        )
        .sum(axis=1)
        .mean()
    )


def evaluate_candidate(
    dataframe: pd.DataFrame,
    *,
    candidate: str,
) -> dict[str, object]:
    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    predicted_home = dataframe[
        "lambda_home"
    ].to_numpy(dtype=float)

    predicted_away = dataframe[
        "lambda_away"
    ].to_numpy(dtype=float)

    home_deviance = mean_poisson_deviance(
        actual_home,
        predicted_home,
    )

    away_deviance = mean_poisson_deviance(
        actual_away,
        predicted_away,
    )

    probabilities = dataframe[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].to_numpy(dtype=float)

    probabilities = np.clip(
        probabilities,
        PROBABILITY_FLOOR,
        1.0,
    )

    probabilities = (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    observed_results = np.array(
        [
            observed_result_index(
                int(home_score),
                int(away_score),
            )
            for home_score, away_score in zip(
                actual_home,
                actual_away,
            )
        ],
        dtype=int,
    )

    exact_score_probabilities = np.array(
        [
            poisson_probability(
                int(home_score),
                float(lambda_home),
            )
            * poisson_probability(
                int(away_score),
                float(lambda_away),
            )
            for (
                home_score,
                away_score,
                lambda_home,
                lambda_away,
            ) in zip(
                actual_home,
                actual_away,
                predicted_home,
                predicted_away,
            )
        ],
        dtype=float,
    )

    actual_draw_rate = float(
        (
            actual_home
            == actual_away
        ).mean()
    )

    predicted_draw_rate = float(
        dataframe[
            "draw_probability"
        ].mean()
    )

    actual_home_mean = float(
        actual_home.mean()
    )

    predicted_home_mean = float(
        predicted_home.mean()
    )

    actual_away_mean = float(
        actual_away.mean()
    )

    predicted_away_mean = float(
        predicted_away.mean()
    )

    return {
        "candidate": candidate,
        "holdout_match_count": len(
            dataframe
        ),
        "artifact_name": (
            dataframe[
                "goal_model_artifact_name"
            ].iloc[0]
        ),
        "artifact_version": (
            dataframe[
                "goal_model_artifact_version"
            ].iloc[0]
        ),
        "feature_specification": (
            dataframe[
                "goal_model_feature_specification"
            ].iloc[0]
        ),
        "training_end_date": (
            dataframe[
                "goal_model_training_end_date"
            ].iloc[0]
        ),
        "home_poisson_deviance": float(
            home_deviance
        ),
        "away_poisson_deviance": float(
            away_deviance
        ),
        "combined_poisson_deviance": float(
            (
                home_deviance
                + away_deviance
            )
            / 2.0
        ),
        "home_goal_mae": float(
            mean_absolute_error(
                actual_home,
                predicted_home,
            )
        ),
        "away_goal_mae": float(
            mean_absolute_error(
                actual_away,
                predicted_away,
            )
        ),
        "combined_goal_mae": float(
            (
                mean_absolute_error(
                    actual_home,
                    predicted_home,
                )
                + mean_absolute_error(
                    actual_away,
                    predicted_away,
                )
            )
            / 2.0
        ),
        "total_goal_mae": float(
            mean_absolute_error(
                actual_home + actual_away,
                predicted_home + predicted_away,
            )
        ),
        "goal_difference_mae": float(
            mean_absolute_error(
                actual_home - actual_away,
                predicted_home - predicted_away,
            )
        ),
        "outcome_log_loss": float(
            log_loss(
                observed_results,
                probabilities,
                labels=[0, 1, 2],
            )
        ),
        "outcome_brier_score":
            multiclass_brier_score(
                observed_results,
                probabilities,
            ),
        "exact_score_log_loss": float(
            -np.log(
                np.clip(
                    exact_score_probabilities,
                    PROBABILITY_FLOOR,
                    1.0,
                )
            ).mean()
        ),
        "actual_draw_rate":
            actual_draw_rate,
        "predicted_draw_rate":
            predicted_draw_rate,
        "absolute_draw_rate_error": float(
            abs(
                predicted_draw_rate
                - actual_draw_rate
            )
        ),
        "actual_home_goal_mean":
            actual_home_mean,
        "predicted_home_goal_mean":
            predicted_home_mean,
        "absolute_home_goal_mean_error": float(
            abs(
                predicted_home_mean
                - actual_home_mean
            )
        ),
        "actual_away_goal_mean":
            actual_away_mean,
        "predicted_away_goal_mean":
            predicted_away_mean,
        "absolute_away_goal_mean_error": float(
            abs(
                predicted_away_mean
                - actual_away_mean
            )
        ),
    }


def build_metric_deltas(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    indexed = performance.set_index(
        "candidate"
    )

    records: list[dict[str, object]] = []

    for metric in COMPARISON_METRICS:
        global_value = float(
            indexed.loc[
                "global_zscore",
                metric,
            ]
        )

        robust_value = float(
            indexed.loc[
                "robust_zscore",
                metric,
            ]
        )

        delta = (
            robust_value
            - global_value
        )

        records.append(
            {
                "metric": metric,
                "global_zscore_value":
                    global_value,
                "robust_zscore_value":
                    robust_value,
                "robust_minus_global":
                    delta,
                "robust_improved":
                    delta < 0.0,
                "tied":
                    bool(
                        np.isclose(
                            delta,
                            0.0,
                            atol=1e-12,
                            rtol=0.0,
                        )
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_paired_fixture_comparison(
    global_replay: pd.DataFrame,
    robust_replay: pd.DataFrame,
) -> pd.DataFrame:
    global_indexed = (
        global_replay
        .set_index("event_id")
        .sort_index()
    )

    robust_indexed = (
        robust_replay
        .set_index("event_id")
        .sort_index()
    )

    output = global_indexed[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    prediction_columns = (
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    )

    for column in prediction_columns:
        output[
            f"global_{column}"
        ] = global_indexed[column]

        output[
            f"robust_{column}"
        ] = robust_indexed[column]

        output[
            f"robust_minus_global_{column}"
        ] = (
            robust_indexed[column]
            - global_indexed[column]
        )

    output[
        "global_combined_absolute_goal_error"
    ] = (
        (
            global_indexed["home_score"]
            - global_indexed["lambda_home"]
        ).abs()
        + (
            global_indexed["away_score"]
            - global_indexed["lambda_away"]
        ).abs()
    ) / 2.0

    output[
        "robust_combined_absolute_goal_error"
    ] = (
        (
            robust_indexed["home_score"]
            - robust_indexed["lambda_home"]
        ).abs()
        + (
            robust_indexed["away_score"]
            - robust_indexed["lambda_away"]
        ).abs()
    ) / 2.0

    output[
        "robust_minus_global_absolute_goal_error"
    ] = (
        output[
            "robust_combined_absolute_goal_error"
        ]
        - output[
            "global_combined_absolute_goal_error"
        ]
    )

    output[
        "absolute_total_lambda_change"
    ] = (
        output[
            "robust_minus_global_lambda_home"
        ].abs()
        + output[
            "robust_minus_global_lambda_away"
        ].abs()
    )

    return output.reset_index()


def determine_preference(
    metric_deltas: pd.DataFrame,
) -> dict[str, object]:
    primary = metric_deltas.loc[
        metric_deltas[
            "metric"
        ].isin(
            PRIMARY_METRICS
        )
    ]

    all_metrics = metric_deltas

    robust_primary_wins = int(
        primary[
            "robust_improved"
        ].sum()
    )

    global_primary_wins = int(
        (
            ~primary["robust_improved"]
            & ~primary["tied"]
        ).sum()
    )

    robust_all_wins = int(
        all_metrics[
            "robust_improved"
        ].sum()
    )

    global_all_wins = int(
        (
            ~all_metrics["robust_improved"]
            & ~all_metrics["tied"]
        ).sum()
    )

    if (
        robust_primary_wins
        > global_primary_wins
    ):
        preferred = "robust_zscore"
    elif (
        global_primary_wins
        > robust_primary_wins
    ):
        preferred = "global_zscore"
    else:
        preferred = "mixed"

    return {
        "preferred_candidate":
            preferred,
        "robust_primary_metric_wins":
            robust_primary_wins,
        "global_primary_metric_wins":
            global_primary_wins,
        "robust_all_metric_wins":
            robust_all_wins,
        "global_all_metric_wins":
            global_all_wins,
    }


def write_report(
    performance: pd.DataFrame,
    metric_deltas: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    indexed = performance.set_index(
        "candidate"
    )

    report = f"""# Study 093C — Production Replay Accuracy Comparison

## Status

**PASS**

## Research question

Does the predictive advantage of `robust_zscore` survive
execution through the complete production observation and
prediction pipeline?

## Evaluation design

- Frozen holdout population: {EXPECTED_HOLDOUT_COUNT} fixtures
- Training fraction: {TRAIN_FRACTION:.2f}
- Candidates:
  - `global_zscore`
  - `robust_zscore`
- Both artifacts trained on the same 229 fixtures
- Both artifacts use the same feature specification
- Both candidates use identical fixture-date ClubElo priors
- Lower values are better for all comparison metrics

## Decision

- Preferred candidate:
  `{decision["preferred_candidate"]}`
- Robust primary-metric wins:
  {decision["robust_primary_metric_wins"]}
- Global primary-metric wins:
  {decision["global_primary_metric_wins"]}
- Robust total metric wins:
  {decision["robust_all_metric_wins"]}
- Global total metric wins:
  {decision["global_all_metric_wins"]}

## Candidate performance

{performance.to_markdown(index=False)}

## Metric deltas

A negative `robust_minus_global` value means that Robust
performed better.

{metric_deltas.to_markdown(index=False)}

## Interpretation boundary

This is a single-season, 77-match chronological holdout.
The comparison validates performance through the production
runtime path, but it does not establish multi-season,
cross-league, or World Cup generalization.

The player representations remain retrospective static
season-level representations. ClubElo priors are fixture-date
valid.

## Validation

- Frozen holdout population: PASS
- Exact candidate event match: PASS
- Exact target match: PASS
- Shared model feature specification: PASS
- Shared model training cutoff: PASS
- Positive finite expected goals: PASS
- Valid outcome probabilities: PASS
- Production replay outputs reused unchanged: PASS
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 093C — PRODUCTION REPLAY "
        "ACCURACY COMPARISON"
    )
    print("=" * 88)

    holdout_event_ids = (
        load_holdout_event_ids()
    )

    global_replay = (
        load_candidate_replay(
            path=GLOBAL_REPLAY_PATH,
            candidate="global_zscore",
            holdout_event_ids=(
                holdout_event_ids
            ),
        )
    )

    robust_replay = (
        load_candidate_replay(
            path=ROBUST_REPLAY_PATH,
            candidate="robust_zscore",
            holdout_event_ids=(
                holdout_event_ids
            ),
        )
    )

    validate_matched_candidates(
        global_replay,
        robust_replay,
    )

    performance = pd.DataFrame(
        [
            evaluate_candidate(
                global_replay,
                candidate="global_zscore",
            ),
            evaluate_candidate(
                robust_replay,
                candidate="robust_zscore",
            ),
        ]
    )

    metric_deltas = (
        build_metric_deltas(
            performance
        )
    )

    paired_fixtures = (
        build_paired_fixture_comparison(
            global_replay,
            robust_replay,
        )
    )

    largest_changes = (
        paired_fixtures
        .sort_values(
            "absolute_total_lambda_change",
            ascending=False,
        )
        .head(25)
        .reset_index(drop=True)
    )

    decision = determine_preference(
        metric_deltas
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance.to_csv(
        PERFORMANCE_PATH,
        index=False,
    )

    metric_deltas.to_csv(
        METRIC_DELTA_PATH,
        index=False,
    )

    paired_fixtures.to_csv(
        PAIRED_FIXTURE_PATH,
        index=False,
    )

    largest_changes.to_csv(
        LARGEST_CHANGES_PATH,
        index=False,
    )

    metadata = {
        "study_id": "093C",
        "study_name": (
            "Production Replay Accuracy Comparison"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "evaluation_population":
            "frozen_chronological_holdout",
        "holdout_match_count":
            EXPECTED_HOLDOUT_COUNT,
        "train_fraction":
            TRAIN_FRACTION,
        "training_match_count":
            229,
        "candidates": [
            "global_zscore",
            "robust_zscore",
        ],
        **decision,
        "matched_event_population":
            True,
        "matched_targets":
            True,
        "shared_feature_specification":
            True,
        "shared_training_cutoff":
            True,
        "production_runtime_path_used":
            True,
        "methodological_boundary": (
            "Single-season 77-match chronological holdout "
            "using retrospective static player-derived "
            "representations."
        ),
        "outputs": [
            PERFORMANCE_PATH.name,
            METRIC_DELTA_PATH.name,
            PAIRED_FIXTURE_PATH.name,
            LARGEST_CHANGES_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        performance=performance,
        metric_deltas=metric_deltas,
        decision=decision,
    )

    display_columns = [
        "candidate",
        *COMPARISON_METRICS,
    ]

    print()
    print("Candidate performance")
    print("-" * 88)
    print(
        performance[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Metric deltas")
    print("-" * 88)
    print(
        metric_deltas.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print(
        "Preferred candidate: "
        f"{decision['preferred_candidate']}"
    )
    print(
        "Primary metric wins — Robust: "
        f"{decision['robust_primary_metric_wins']}; "
        "Global: "
        f"{decision['global_primary_metric_wins']}"
    )

    print()
    print("Validation summary")
    print("  Holdout population loading: PASS")
    print("  Candidate event matching: PASS")
    print("  Candidate target matching: PASS")
    print("  Shared feature specification: PASS")
    print("  Shared training cutoff: PASS")
    print("  Finite expected goals: PASS")
    print("  Outcome probabilities: PASS")
    print("  Metric calculation: PASS")
    print("  Output generation: PASS")

    print()
    print("=" * 88)
    print("OVERALL EXECUTION RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()