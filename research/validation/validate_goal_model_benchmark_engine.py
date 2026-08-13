#validate_goal_model_benchmark_engine

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.benchmarking.goal_model_benchmark import (
    GoalModelBenchmarkConfig,
    GoalModelDatasetConfig,
    run_goal_model_benchmark,
)
from research.modeling.football_feature_registry import (
    list_club_goal_model_feature_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

ARCHIVED_STUDY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_052_club_feature_ablation"
    / "feature_ablation_results.csv"
)


ALPHA_VALUES = (
    0.0,
    0.01,
    0.1,
    1.0,
)

TRAIN_FRACTION = 0.75

COMPARISON_TOLERANCE = 1e-10


METRIC_COLUMNS = [
    "home_poisson_deviance",
    "away_poisson_deviance",
    "combined_poisson_deviance",
    "home_goal_mae",
    "away_goal_mae",
    "combined_goal_mae",
    "total_goal_mae",
    "goal_difference_mae",
    "outcome_log_loss",
    "outcome_brier_score",
    "exact_score_log_loss",
    "actual_draw_rate",
    "predicted_draw_rate",
    "draw_rate_error",
    "absolute_draw_rate_error",
    "actual_home_goal_mean",
    "predicted_home_goal_mean",
    "home_goal_mean_error",
    "absolute_home_goal_mean_error",
    "actual_away_goal_mean",
    "predicted_away_goal_mean",
    "away_goal_mean_error",
    "absolute_away_goal_mean_error",
]


def load_archived_results() -> pd.DataFrame:
    if not ARCHIVED_STUDY_PATH.exists():
        raise FileNotFoundError(
            "Archived Study 052 results do not exist: "
            f"{ARCHIVED_STUDY_PATH}"
        )

    dataframe = pd.read_csv(
        ARCHIVED_STUDY_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Archived Study 052 results are empty."
        )

    required_columns = {
        "specification",
        "alpha",
        *METRIC_COLUMNS,
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Archived Study 052 results are "
            f"missing columns: {sorted(missing)}"
        )

    return dataframe


def run_reproduction_benchmark() -> pd.DataFrame:
    config = GoalModelBenchmarkConfig(
        name=(
            "study_052_engine_reproduction"
        ),
        datasets=(
            GoalModelDatasetConfig(
                name="full_squad",
                path=OBSERVATION_PATH,
                representation_type=(
                    "full_squad"
                ),
            ),
        ),
        feature_specifications=(
            list_club_goal_model_feature_specs()
        ),
        train_fractions=(
            TRAIN_FRACTION,
        ),
        alpha_values=ALPHA_VALUES,
        capture_predictions=False,
        capture_coefficients=False,
    )

    benchmark_result = (
        run_goal_model_benchmark(
            config
        )
    )

    return benchmark_result.results


def compare_results(
    archived: pd.DataFrame,
    reproduced: pd.DataFrame,
) -> pd.DataFrame:
    archived = archived.rename(
        columns={
            "specification":
                "feature_specification",
        }
    )

    key_columns = [
        "feature_specification",
        "alpha",
    ]

    archived_indexed = (
        archived
        .set_index(key_columns)
        .sort_index()
    )

    reproduced_indexed = (
        reproduced
        .set_index(key_columns)
        .sort_index()
    )

    if not archived_indexed.index.equals(
        reproduced_indexed.index
    ):
        raise AssertionError(
            "Archived and reproduced configuration "
            "populations do not match."
        )

    comparison_rows: list[
        dict[str, object]
    ] = []

    for configuration_key in (
        archived_indexed.index
    ):
        archived_row = (
            archived_indexed.loc[
                configuration_key
            ]
        )

        reproduced_row = (
            reproduced_indexed.loc[
                configuration_key
            ]
        )

        row: dict[str, object] = {
            "feature_specification":
                configuration_key[0],
            "alpha":
                configuration_key[1],
        }

        configuration_pass = True
        maximum_difference = 0.0

        for metric in METRIC_COLUMNS:
            archived_value = float(
                archived_row[metric]
            )

            reproduced_value = float(
                reproduced_row[metric]
            )

            difference = abs(
                archived_value
                - reproduced_value
            )

            row[
                f"{metric}_archived"
            ] = archived_value

            row[
                f"{metric}_reproduced"
            ] = reproduced_value

            row[
                f"{metric}_absolute_difference"
            ] = difference

            maximum_difference = max(
                maximum_difference,
                difference,
            )

            if not np.isclose(
                archived_value,
                reproduced_value,
                atol=COMPARISON_TOLERANCE,
                rtol=0.0,
                equal_nan=True,
            ):
                configuration_pass = False

        row[
            "maximum_absolute_difference"
        ] = maximum_difference

        row[
            "validation_pass"
        ] = configuration_pass

        comparison_rows.append(row)

    return pd.DataFrame(
        comparison_rows
    )


def main() -> None:
    archived = load_archived_results()

    reproduced = (
        run_reproduction_benchmark()
    )

    comparison = compare_results(
        archived=archived,
        reproduced=reproduced,
    )

    failures = comparison[
        ~comparison["validation_pass"]
    ]

    if not failures.empty:
        print(
            failures[
                [
                    "feature_specification",
                    "alpha",
                    "maximum_absolute_difference",
                ]
            ]
            .sort_values(
                "maximum_absolute_difference",
                ascending=False,
            )
            .to_string(index=False)
        )

        raise AssertionError(
            "Benchmark-engine reproduction "
            "validation failed."
        )

    print(
        "Goal Model Benchmark Engine Validation"
    )
    print("=" * 76)
    print()
    print(
        "Archived configurations: "
        f"{len(archived)}"
    )
    print(
        "Reproduced configurations: "
        f"{len(reproduced)}"
    )
    print(
        "Compared metrics per configuration: "
        f"{len(METRIC_COLUMNS)}"
    )
    print(
        "Maximum absolute difference: "
        f"{comparison['maximum_absolute_difference'].max():.12e}"
    )
    print()

    print("Configuration population: PASS")
    print("Metric reproduction: PASS")
    print("Chronological split reproduction: PASS")
    print("Feature-registry integration: PASS")
    print("Poisson-model integration: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()