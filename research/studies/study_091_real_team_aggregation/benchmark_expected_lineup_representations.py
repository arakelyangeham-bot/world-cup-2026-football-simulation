#benchmark_expected_lineup_representations.py

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_091_real_team_aggregation"
)

OUTPUT_DIRECTORY = INPUT_DIRECTORY

PLAYER_POPULATION_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_player_population.csv"
)

TEAM_COVERAGE_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_team_coverage.csv"
)

COMPATIBILITY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_real_scale_compatibility.csv"
)

SPECIFICATION_SUMMARY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_real_scale_specification_summary.csv"
)

TEAM_REPRESENTATION_LONG_PATH = (
    OUTPUT_DIRECTORY
    / "expected_lineup_team_representations_long.csv"
)

TEAM_REPRESENTATION_WIDE_PATH = (
    OUTPUT_DIRECTORY
    / "expected_lineup_team_representations_wide.csv"
)

DIMENSION_RANK_PATH = (
    OUTPUT_DIRECTORY
    / "expected_lineup_dimension_rankings.csv"
)

RANK_CORRELATION_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_rank_correlations.csv"
)

DISAGREEMENT_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_team_disagreements.csv"
)

SPECIFICATION_PROFILE_PATH = (
    OUTPUT_DIRECTORY
    / "real_team_aggregation_profiles.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_091b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_091b_report.md"
)


BASELINE_SPECIFICATION_ID = "top5_arithmetic"

EXCLUDED_SPECIFICATION_IDS = {
    "top5_power_1_50": (
        "Excluded because Study 091A found a real-scale domain "
        "failure on negative-valued player projections."
    ),
}


TEAM_COMPOSITE_WEIGHTS = {
    "attack": 0.35,
    "midfield": 0.25,
    "defense": 0.25,
    "goalkeeper": 0.15,
}


REQUIRED_COMPATIBILITY_COLUMNS = {
    "country",
    "formation",
    "dimension",
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "parameterization",
    "historical_control",
    "status",
    "aggregated_value",
}


REQUIRED_POPULATION_COLUMNS = {
    "country",
    "formation",
    "player_id",
    "projected_attack",
    "projected_midfield",
    "projected_defense",
    "projected_goalkeeper",
}


REQUIRED_COVERAGE_COLUMNS = {
    "country",
    "complete_expected_xi",
    "eligible_for_real_scale_audit",
}


REQUIRED_SPECIFICATION_SUMMARY_COLUMNS = {
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "parameterization",
    "historical_control",
    "failed_rows",
    "real_scale_compatible",
}


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(
        required_columns - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}"
        )


def coerce_boolean(
    values: pd.Series,
) -> pd.Series:
    if values.dtype == bool:
        return values

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    result = normalized.map(mapping)

    if result.isna().any():
        invalid = sorted(
            normalized.loc[
                result.isna()
            ].unique()
        )

        raise ValueError(
            "Could not coerce boolean values: "
            f"{invalid}"
        )

    return result.astype(bool)


def load_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    player_population = pd.read_csv(
        PLAYER_POPULATION_PATH
    )

    team_coverage = pd.read_csv(
        TEAM_COVERAGE_PATH
    )

    compatibility = pd.read_csv(
        COMPATIBILITY_PATH
    )

    specification_summary = pd.read_csv(
        SPECIFICATION_SUMMARY_PATH
    )

    require_columns(
        player_population,
        REQUIRED_POPULATION_COLUMNS,
        frame_name="Expected-lineup player population",
    )

    require_columns(
        team_coverage,
        REQUIRED_COVERAGE_COLUMNS,
        frame_name="Expected-lineup team coverage",
    )

    require_columns(
        compatibility,
        REQUIRED_COMPATIBILITY_COLUMNS,
        frame_name="Real-scale compatibility results",
    )

    require_columns(
        specification_summary,
        REQUIRED_SPECIFICATION_SUMMARY_COLUMNS,
        frame_name="Real-scale specification summary",
    )

    team_coverage = team_coverage.copy()
    specification_summary = (
        specification_summary.copy()
    )

    team_coverage[
        "complete_expected_xi"
    ] = coerce_boolean(
        team_coverage[
            "complete_expected_xi"
        ]
    )

    team_coverage[
        "eligible_for_real_scale_audit"
    ] = coerce_boolean(
        team_coverage[
            "eligible_for_real_scale_audit"
        ]
    )

    specification_summary[
        "real_scale_compatible"
    ] = coerce_boolean(
        specification_summary[
            "real_scale_compatible"
        ]
    )

    return (
        player_population,
        team_coverage,
        compatibility,
        specification_summary,
    )


def select_compatible_specifications(
    specification_summary: pd.DataFrame,
) -> pd.DataFrame:
    selected = specification_summary.loc[
        specification_summary[
            "real_scale_compatible"
        ]
    ].copy()

    selected = selected.loc[
        ~selected[
            "specification_id"
        ].isin(
            EXCLUDED_SPECIFICATION_IDS
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No real-scale-compatible aggregation "
            "specifications are available."
        )

    if (
        BASELINE_SPECIFICATION_ID
        not in set(
            selected[
                "specification_id"
            ]
        )
    ):
        raise ValueError(
            "The baseline specification "
            f"{BASELINE_SPECIFICATION_ID!r} is missing."
        )

    if selected[
        "specification_id"
    ].duplicated().any():
        raise ValueError(
            "Compatible specification table contains "
            "duplicate specification IDs."
        )

    return (
        selected
        .sort_values("specification_id")
        .reset_index(drop=True)
    )


def build_goalkeeper_values(
    *,
    player_population: pd.DataFrame,
    eligible_teams: set[str],
) -> pd.DataFrame:
    selected = player_population.loc[
        player_population[
            "country"
        ].isin(eligible_teams)
    ].copy()

    selected[
        "projected_goalkeeper"
    ] = pd.to_numeric(
        selected[
            "projected_goalkeeper"
        ],
        errors="coerce",
    )

    if selected[
        "projected_goalkeeper"
    ].isna().any():
        raise ValueError(
            "Eligible player population contains missing "
            "goalkeeper projections."
        )

    values = (
        selected
        .groupby(
            "country",
            as_index=False,
        )
        .agg(
            goalkeeper=(
                "projected_goalkeeper",
                "max",
            ),
            goalkeeper_population_mean=(
                "projected_goalkeeper",
                "mean",
            ),
            expected_xi_player_count=(
                "player_id",
                "size",
            ),
            formation=(
                "formation",
                "first",
            ),
        )
    )

    invalid_counts = values.loc[
        ~values[
            "expected_xi_player_count"
        ].eq(11)
    ]

    if not invalid_counts.empty:
        raise AssertionError(
            "One or more eligible teams do not contain "
            "exactly 11 players."
        )

    return values


def build_team_representations(
    *,
    compatibility: pd.DataFrame,
    selected_specifications: pd.DataFrame,
    goalkeeper_values: pd.DataFrame,
    eligible_teams: set[str],
) -> pd.DataFrame:
    specification_ids = set(
        selected_specifications[
            "specification_id"
        ]
    )

    selected = compatibility.loc[
        compatibility[
            "country"
        ].isin(eligible_teams)
        & compatibility[
            "specification_id"
        ].isin(specification_ids)
    ].copy()

    failed = selected.loc[
        selected[
            "status"
        ].ne("evaluated")
    ]

    if not failed.empty:
        raise ValueError(
            "Study 091B received non-evaluated rows for "
            "supposedly compatible specifications."
        )

    selected[
        "aggregated_value"
    ] = pd.to_numeric(
        selected[
            "aggregated_value"
        ],
        errors="coerce",
    )

    if selected[
        "aggregated_value"
    ].isna().any():
        raise ValueError(
            "Compatible aggregation rows contain missing values."
        )

    if not np.isfinite(
        selected[
            "aggregated_value"
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Compatible aggregation rows contain non-finite values."
        )

    dimension_wide = (
        selected
        .pivot(
            index=[
                "country",
                "specification_id",
                "aggregation_family",
                "display_name",
                "output_type",
                "parameterization",
                "historical_control",
            ],
            columns="dimension",
            values="aggregated_value",
        )
        .reset_index()
    )

    dimension_wide.columns.name = None

    required_dimensions = {
        "attack",
        "midfield",
        "defense",
    }

    missing_dimensions = (
        required_dimensions
        - set(dimension_wide.columns)
    )

    if missing_dimensions:
        raise ValueError(
            "Team representation table is missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    representations = dimension_wide.merge(
        goalkeeper_values[
            [
                "country",
                "goalkeeper",
                "goalkeeper_population_mean",
                "expected_xi_player_count",
                "formation",
            ]
        ],
        on="country",
        how="left",
        validate="many_to_one",
    )

    if representations[
        "goalkeeper"
    ].isna().any():
        raise ValueError(
            "One or more team representations are missing "
            "goalkeeper values."
        )

    representations[
        "outfield_mean"
    ] = (
        representations[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ]
        .mean(axis=1)
    )

    representations[
        "team_composite"
    ] = (
        representations["attack"]
        * TEAM_COMPOSITE_WEIGHTS["attack"]
        + representations["midfield"]
        * TEAM_COMPOSITE_WEIGHTS["midfield"]
        + representations["defense"]
        * TEAM_COMPOSITE_WEIGHTS["defense"]
        + representations["goalkeeper"]
        * TEAM_COMPOSITE_WEIGHTS["goalkeeper"]
    )

    representations[
        "outfield_balance_range"
    ] = (
        representations[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ].max(axis=1)
        - representations[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ].min(axis=1)
    )

    representations[
        "strongest_outfield_dimension"
    ] = (
        representations[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ].idxmax(axis=1)
    )

    representations[
        "weakest_outfield_dimension"
    ] = (
        representations[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ].idxmin(axis=1)
    )

    return (
        representations
        .sort_values(
            [
                "specification_id",
                "team_composite",
                "country",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def add_within_specification_ranks(
    representations: pd.DataFrame,
) -> pd.DataFrame:
    output = representations.copy()

    rank_columns = {
        "attack": "attack_rank",
        "midfield": "midfield_rank",
        "defense": "defense_rank",
        "goalkeeper": "goalkeeper_rank",
        "outfield_mean": "outfield_rank",
        "team_composite": "team_composite_rank",
    }

    for value_column, rank_column in (
        rank_columns.items()
    ):
        output[rank_column] = (
            output
            .groupby(
                "specification_id"
            )[value_column]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

    return output


def build_long_representation_output(
    ranked_representations: pd.DataFrame,
) -> pd.DataFrame:
    dimensions = (
        (
            "attack",
            "attack",
            "attack_rank",
        ),
        (
            "midfield",
            "midfield",
            "midfield_rank",
        ),
        (
            "defense",
            "defense",
            "defense_rank",
        ),
        (
            "goalkeeper",
            "goalkeeper",
            "goalkeeper_rank",
        ),
        (
            "outfield_mean",
            "outfield_mean",
            "outfield_rank",
        ),
        (
            "team_composite",
            "team_composite",
            "team_composite_rank",
        ),
    )

    records: list[dict[str, object]] = []

    for row in ranked_representations.itertuples(
        index=False
    ):
        for (
            representation_dimension,
            value_column,
            rank_column,
        ) in dimensions:
            records.append(
                {
                    "country":
                        row.country,
                    "formation":
                        row.formation,
                    "specification_id":
                        row.specification_id,
                    "aggregation_family":
                        row.aggregation_family,
                    "display_name":
                        row.display_name,
                    "output_type":
                        row.output_type,
                    "parameterization":
                        row.parameterization,
                    "historical_control":
                        row.historical_control,
                    "representation_dimension":
                        representation_dimension,
                    "representation_value":
                        float(
                            getattr(
                                row,
                                value_column,
                            )
                        ),
                    "within_specification_rank":
                        int(
                            getattr(
                                row,
                                rank_column,
                            )
                        ),
                    "expected_xi_player_count":
                        int(
                            row.expected_xi_player_count
                        ),
                }
            )

    return pd.DataFrame(records)


def build_dimension_rankings(
    long_representations: pd.DataFrame,
) -> pd.DataFrame:
    return (
        long_representations
        .sort_values(
            [
                "representation_dimension",
                "specification_id",
                "within_specification_rank",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def build_rank_correlations(
    long_representations: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for dimension, dimension_rows in (
        long_representations.groupby(
            "representation_dimension",
            sort=True,
        )
    ):
        value_matrix = (
            dimension_rows
            .pivot(
                index="country",
                columns="specification_id",
                values="representation_value",
            )
        )

        rank_matrix = (
            dimension_rows
            .pivot(
                index="country",
                columns="specification_id",
                values="within_specification_rank",
            )
        )

        specification_ids = sorted(
            value_matrix.columns
        )

        for left_index, left_id in enumerate(
            specification_ids
        ):
            for right_id in specification_ids[
                left_index:
            ]:
                pearson = value_matrix[
                    left_id
                ].corr(
                    value_matrix[
                        right_id
                    ],
                    method="pearson",
                )

                spearman = rank_matrix[
                    left_id
                ].corr(
                    rank_matrix[
                        right_id
                    ],
                    method="pearson",
                )

                records.append(
                    {
                        "representation_dimension":
                            dimension,
                        "specification_id_a":
                            left_id,
                        "specification_id_b":
                            right_id,
                        "team_count":
                            len(value_matrix),
                        "pearson_value_correlation":
                            float(pearson),
                        "spearman_rank_correlation":
                            float(spearman),
                        "absolute_rank_correlation_gap":
                            float(
                                abs(
                                    1.0
                                    - spearman
                                )
                            ),
                    }
                )

    return pd.DataFrame(records)


def build_team_disagreements(
    long_representations: pd.DataFrame,
) -> pd.DataFrame:
    baseline = long_representations.loc[
        long_representations[
            "specification_id"
        ].eq(BASELINE_SPECIFICATION_ID)
    ][
        [
            "country",
            "representation_dimension",
            "representation_value",
            "within_specification_rank",
        ]
    ].rename(
        columns={
            "representation_value":
                "baseline_value",
            "within_specification_rank":
                "baseline_rank",
        }
    )

    alternatives = long_representations.loc[
        ~long_representations[
            "specification_id"
        ].eq(BASELINE_SPECIFICATION_ID)
    ].copy()

    disagreements = alternatives.merge(
        baseline,
        on=[
            "country",
            "representation_dimension",
        ],
        how="left",
        validate="many_to_one",
    )

    disagreements[
        "value_difference_vs_baseline"
    ] = (
        disagreements[
            "representation_value"
        ]
        - disagreements[
            "baseline_value"
        ]
    )

    disagreements[
        "absolute_value_difference_vs_baseline"
    ] = disagreements[
        "value_difference_vs_baseline"
    ].abs()

    disagreements[
        "rank_change_vs_baseline"
    ] = (
        disagreements[
            "baseline_rank"
        ]
        - disagreements[
            "within_specification_rank"
        ]
    )

    disagreements[
        "absolute_rank_change_vs_baseline"
    ] = disagreements[
        "rank_change_vs_baseline"
    ].abs()

    disagreements[
        "direction_vs_baseline"
    ] = np.select(
        [
            disagreements[
                "value_difference_vs_baseline"
            ].gt(1e-12),
            disagreements[
                "value_difference_vs_baseline"
            ].lt(-1e-12),
        ],
        [
            "higher",
            "lower",
        ],
        default="equal",
    )

    return (
        disagreements
        .sort_values(
            [
                "representation_dimension",
                "absolute_rank_change_vs_baseline",
                "absolute_value_difference_vs_baseline",
                "country",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_specification_profiles(
    *,
    ranked_representations: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    baseline_correlations = (
        rank_correlations.loc[
            rank_correlations[
                "specification_id_a"
            ].eq(BASELINE_SPECIFICATION_ID)
            | rank_correlations[
                "specification_id_b"
            ].eq(BASELINE_SPECIFICATION_ID)
        ]
        .copy()
    )

    for specification_id, rows in (
        ranked_representations.groupby(
            "specification_id",
            sort=True,
        )
    ):
        metadata_row = rows.iloc[0]

        specification_disagreements = (
            disagreements.loc[
                disagreements[
                    "specification_id"
                ].eq(specification_id)
            ]
        )

        if (
            specification_id
            == BASELINE_SPECIFICATION_ID
        ):
            mean_absolute_rank_change = 0.0
            maximum_absolute_rank_change = 0
            mean_absolute_value_difference = 0.0
            mean_spearman_vs_baseline = 1.0
        else:
            mean_absolute_rank_change = float(
                specification_disagreements[
                    "absolute_rank_change_vs_baseline"
                ].mean()
            )

            maximum_absolute_rank_change = int(
                specification_disagreements[
                    "absolute_rank_change_vs_baseline"
                ].max()
            )

            mean_absolute_value_difference = float(
                specification_disagreements[
                    "absolute_value_difference_vs_baseline"
                ].mean()
            )

            correlation_rows = (
                baseline_correlations.loc[
                    (
                        baseline_correlations[
                            "specification_id_a"
                        ].eq(
                            BASELINE_SPECIFICATION_ID
                        )
                        & baseline_correlations[
                            "specification_id_b"
                        ].eq(specification_id)
                    )
                    | (
                        baseline_correlations[
                            "specification_id_b"
                        ].eq(
                            BASELINE_SPECIFICATION_ID
                        )
                        & baseline_correlations[
                            "specification_id_a"
                        ].eq(specification_id)
                    )
                ]
            )

            mean_spearman_vs_baseline = float(
                correlation_rows[
                    "spearman_rank_correlation"
                ].mean()
            )

        top_composite_teams = (
            rows
            .sort_values(
                [
                    "team_composite_rank",
                    "country",
                ]
            )
            .head(10)[
                "country"
            ]
            .tolist()
        )

        records.append(
            {
                "specification_id":
                    specification_id,
                "aggregation_family":
                    metadata_row[
                        "aggregation_family"
                    ],
                "display_name":
                    metadata_row[
                        "display_name"
                    ],
                "output_type":
                    metadata_row[
                        "output_type"
                    ],
                "parameterization":
                    metadata_row[
                        "parameterization"
                    ],
                "historical_control":
                    metadata_row[
                        "historical_control"
                    ],
                "team_count":
                    rows[
                        "country"
                    ].nunique(),
                "mean_attack":
                    float(
                        rows[
                            "attack"
                        ].mean()
                    ),
                "mean_midfield":
                    float(
                        rows[
                            "midfield"
                        ].mean()
                    ),
                "mean_defense":
                    float(
                        rows[
                            "defense"
                        ].mean()
                    ),
                "mean_team_composite":
                    float(
                        rows[
                            "team_composite"
                        ].mean()
                    ),
                "team_composite_standard_deviation":
                    float(
                        rows[
                            "team_composite"
                        ].std(ddof=0)
                    ),
                "mean_absolute_rank_change_vs_baseline":
                    mean_absolute_rank_change,
                "maximum_absolute_rank_change_vs_baseline":
                    maximum_absolute_rank_change,
                "mean_absolute_value_difference_vs_baseline":
                    mean_absolute_value_difference,
                "mean_spearman_rank_correlation_vs_baseline":
                    mean_spearman_vs_baseline,
                "top_10_composite_teams":
                    json.dumps(
                        top_composite_teams
                    ),
            }
        )

    return pd.DataFrame(records)


def validate_outputs(
    *,
    selected_specifications: pd.DataFrame,
    eligible_teams: set[str],
    wide_representations: pd.DataFrame,
    long_representations: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
    specification_profiles: pd.DataFrame,
) -> None:
    specification_count = len(
        selected_specifications
    )
    team_count = len(
        eligible_teams
    )

    expected_wide_rows = (
        specification_count
        * team_count
    )

    expected_long_rows = (
        expected_wide_rows
        * 6
    )

    if len(
        wide_representations
    ) != expected_wide_rows:
        raise AssertionError(
            "Unexpected wide representation row count. "
            f"Expected {expected_wide_rows}, "
            f"received {len(wide_representations)}."
        )

    if len(
        long_representations
    ) != expected_long_rows:
        raise AssertionError(
            "Unexpected long representation row count. "
            f"Expected {expected_long_rows}, "
            f"received {len(long_representations)}."
        )

    if wide_representations[
        [
            "country",
            "specification_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Wide representation output contains duplicate "
            "team-specification rows."
        )

    if long_representations[
        [
            "country",
            "specification_id",
            "representation_dimension",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Long representation output contains duplicate rows."
        )

    if len(
        specification_profiles
    ) != specification_count:
        raise AssertionError(
            "Specification profile output does not cover "
            "all selected specifications."
        )

    if set(
        specification_profiles[
            "specification_id"
        ]
    ) != set(
        selected_specifications[
            "specification_id"
        ]
    ):
        raise AssertionError(
            "Specification profiles do not match selected registry."
        )

    numeric_columns = [
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "outfield_mean",
        "team_composite",
        "outfield_balance_range",
    ]

    values = wide_representations[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise AssertionError(
            "Team representations contain non-finite values."
        )

    if rank_correlations[
        "spearman_rank_correlation"
    ].isna().any():
        raise AssertionError(
            "Rank-correlation output contains missing values."
        )

    if disagreements.empty:
        raise AssertionError(
            "Disagreement output is empty."
        )


def build_metadata(
    *,
    selected_specifications: pd.DataFrame,
    eligible_teams: set[str],
    wide_representations: pd.DataFrame,
    long_representations: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> dict[str, Any]:
    maximum_rank_change = int(
        disagreements[
            "absolute_rank_change_vs_baseline"
        ].max()
    )

    mean_rank_change = float(
        disagreements[
            "absolute_rank_change_vs_baseline"
        ].mean()
    )

    return {
        "study_id": "091B",
        "study_name": (
            "Expected-Lineup National-Team "
            "Representation Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "team_count":
            len(eligible_teams),
        "aggregation_specification_count":
            len(selected_specifications),
        "team_representation_row_count":
            len(wide_representations),
        "long_representation_row_count":
            len(long_representations),
        "rank_correlation_row_count":
            len(rank_correlations),
        "disagreement_row_count":
            len(disagreements),
        "baseline_specification_id":
            BASELINE_SPECIFICATION_ID,
        "excluded_specifications":
            EXCLUDED_SPECIFICATION_IDS,
        "maximum_absolute_rank_change_vs_baseline":
            maximum_rank_change,
        "mean_absolute_rank_change_vs_baseline":
            mean_rank_change,
        "team_composite_weights":
            TEAM_COMPOSITE_WEIGHTS,
        "ranking_scope": (
            "Descriptive within-specification team ranks only."
        ),
        "overall_aggregation_ranking_generated":
            False,
        "winning_aggregation_declared":
            False,
        "team_repository_generated":
            False,
        "goal_model_fitted":
            False,
        "match_outcomes_used":
            False,
        "production_repository_changed":
            False,
        "production_runtime_changed":
            False,
        "population_contract": (
            "Complete expected 4-3-3 starting XIs from Study 091A."
        ),
        "interpretation_boundary": (
            "Team rankings and disagreements describe how national-team "
            "representations change under different aggregation "
            "philosophies. They do not establish predictive superiority."
        ),
        "outputs": [
            TEAM_REPRESENTATION_LONG_PATH.name,
            TEAM_REPRESENTATION_WIDE_PATH.name,
            DIMENSION_RANK_PATH.name,
            RANK_CORRELATION_PATH.name,
            DISAGREEMENT_PATH.name,
            SPECIFICATION_PROFILE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    selected_specifications: pd.DataFrame,
    ranked_representations: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
    specification_profiles: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    alternative_disagreements = (
        disagreements.loc[
            disagreements[
                "representation_dimension"
            ].eq("team_composite")
        ]
        .sort_values(
            [
                "absolute_rank_change_vs_baseline",
                "absolute_value_difference_vs_baseline",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(20)[
            [
                "country",
                "specification_id",
                "baseline_rank",
                "within_specification_rank",
                "rank_change_vs_baseline",
                "value_difference_vs_baseline",
            ]
        ]
        .to_dict(orient="records")
    )

    composite_correlations = (
        rank_correlations.loc[
            rank_correlations[
                "representation_dimension"
            ].eq("team_composite")
            & ~rank_correlations[
                "specification_id_a"
            ].eq(
                rank_correlations[
                    "specification_id_b"
                ]
            )
        ]
        .sort_values(
            "spearman_rank_correlation"
        )
        .head(15)[
            [
                "specification_id_a",
                "specification_id_b",
                "spearman_rank_correlation",
                "pearson_value_correlation",
            ]
        ]
        .to_dict(orient="records")
    )

    top_teams_by_specification: dict[
        str,
        list[str],
    ] = {}

    for specification_id, rows in (
        ranked_representations.groupby(
            "specification_id"
        )
    ):
        top_teams_by_specification[
            specification_id
        ] = (
            rows
            .sort_values(
                [
                    "team_composite_rank",
                    "country",
                ]
            )
            .head(10)[
                "country"
            ]
            .tolist()
        )

    excluded_text = json.dumps(
        EXCLUDED_SPECIFICATION_IDS,
        indent=2,
    )

    report = f"""# Study 091B — Expected-Lineup National-Team Representation Benchmark

## Purpose

Compare real national-team representations under the aggregation
philosophies that passed Study 091A's real-scale compatibility audit.

## Population contract

- Teams: {metadata["team_count"]}
- Population: complete expected 4-3-3 starting XI
- Players per team: 11
- Aggregation specifications:
  {metadata["aggregation_specification_count"]}
- Baseline specification:
  `{BASELINE_SPECIFICATION_ID}`

## Included aggregation specifications

{json.dumps(
    selected_specifications["specification_id"].tolist(),
    indent=2,
)}

## Excluded aggregation specifications

{excluded_text}

## Representation dimensions

Each aggregation specification produces:

- attack;
- midfield;
- defense;
- outfield mean;
- existing maximum-goalkeeper representation;
- descriptive team composite.

The team composite uses the project's existing dimensional weights:

{json.dumps(TEAM_COMPOSITE_WEIGHTS, indent=2)}

The composite is descriptive. It is not a fitted goal-model coefficient
and does not constitute a production repository.

## Top ten teams by specification

{json.dumps(top_teams_by_specification, indent=2)}

## Largest composite-rank changes from the baseline

{json.dumps(alternative_disagreements, indent=2)}

## Lowest cross-method composite-rank correlations

{json.dumps(composite_correlations, indent=2)}

## Specification-level profiles

{json.dumps(
    specification_profiles.to_dict(orient="records"),
    indent=2,
    default=str,
)}

## Interpretation

Large positive rank changes identify teams that benefit from the
assumptions encoded by an alternative aggregation.

Examples:

- rank-weighted and star-influence methods may reward teams with a
  stronger elite core;
- whole-population arithmetic may reward balanced expected XIs;
- ability-power weighting may respond to the full distribution while
  emphasizing superior players;
- softmax may produce stronger separation among top-heavy teams.

These are descriptive hypotheses, not predictive conclusions.

## Methodological boundary

This study:

- does not fit a goal model;
- does not use match outcomes;
- does not generate production repositories;
- does not change `TEAM_REPOSITORY_SOURCE`;
- does not declare a winning aggregation;
- does not transform the player-rating scale;
- does not reintroduce the incompatible power-mean candidate.

## Result

**OVERALL RESULT: {metadata["status"]}**

Study 091B generated real national-team representation comparisons
without altering production behavior.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 091B — EXPECTED-LINEUP NATIONAL-TEAM "
        "REPRESENTATION BENCHMARK"
    )
    print("=" * 88)

    (
        player_population,
        team_coverage,
        compatibility,
        specification_summary,
    ) = load_inputs()

    selected_specifications = (
        select_compatible_specifications(
            specification_summary
        )
    )

    eligible_teams = set(
        team_coverage.loc[
            team_coverage[
                "complete_expected_xi"
            ]
            & team_coverage[
                "eligible_for_real_scale_audit"
            ],
            "country",
        ]
    )

    if not eligible_teams:
        raise ValueError(
            "No complete expected XIs are available."
        )

    goalkeeper_values = (
        build_goalkeeper_values(
            player_population=player_population,
            eligible_teams=eligible_teams,
        )
    )

    representations = (
        build_team_representations(
            compatibility=compatibility,
            selected_specifications=(
                selected_specifications
            ),
            goalkeeper_values=goalkeeper_values,
            eligible_teams=eligible_teams,
        )
    )

    ranked_representations = (
        add_within_specification_ranks(
            representations
        )
    )

    long_representations = (
        build_long_representation_output(
            ranked_representations
        )
    )

    dimension_rankings = (
        build_dimension_rankings(
            long_representations
        )
    )

    rank_correlations = (
        build_rank_correlations(
            long_representations
        )
    )

    disagreements = (
        build_team_disagreements(
            long_representations
        )
    )

    specification_profiles = (
        build_specification_profiles(
            ranked_representations=(
                ranked_representations
            ),
            rank_correlations=rank_correlations,
            disagreements=disagreements,
        )
    )

    validate_outputs(
        selected_specifications=(
            selected_specifications
        ),
        eligible_teams=eligible_teams,
        wide_representations=(
            ranked_representations
        ),
        long_representations=(
            long_representations
        ),
        rank_correlations=rank_correlations,
        disagreements=disagreements,
        specification_profiles=(
            specification_profiles
        ),
    )

    metadata = build_metadata(
        selected_specifications=(
            selected_specifications
        ),
        eligible_teams=eligible_teams,
        wide_representations=(
            ranked_representations
        ),
        long_representations=(
            long_representations
        ),
        rank_correlations=rank_correlations,
        disagreements=disagreements,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked_representations.to_csv(
        TEAM_REPRESENTATION_WIDE_PATH,
        index=False,
    )

    long_representations.to_csv(
        TEAM_REPRESENTATION_LONG_PATH,
        index=False,
    )

    dimension_rankings.to_csv(
        DIMENSION_RANK_PATH,
        index=False,
    )

    rank_correlations.to_csv(
        RANK_CORRELATION_PATH,
        index=False,
    )

    disagreements.to_csv(
        DISAGREEMENT_PATH,
        index=False,
    )

    specification_profiles.to_csv(
        SPECIFICATION_PROFILE_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        selected_specifications=(
            selected_specifications
        ),
        ranked_representations=(
            ranked_representations
        ),
        rank_correlations=rank_correlations,
        disagreements=disagreements,
        specification_profiles=(
            specification_profiles
        ),
        metadata=metadata,
    )

    print()
    print("Benchmark coverage")
    print("-" * 88)
    print(
        f"  Complete expected XIs: "
        f"{len(eligible_teams)}"
    )
    print(
        "  Compatible aggregation specifications: "
        f"{len(selected_specifications)}"
    )
    print(
        "  Team-specification representations: "
        f"{len(ranked_representations)}"
    )
    print(
        "  Long representation rows: "
        f"{len(long_representations)}"
    )

    print()
    print("Included specifications")
    print("-" * 88)
    print(
        selected_specifications[
            [
                "specification_id",
                "aggregation_family",
                "historical_control",
            ]
        ].to_string(index=False)
    )

    print()
    print("Specification profiles")
    print("-" * 88)
    print(
        specification_profiles[
            [
                "specification_id",
                "mean_team_composite",
                "team_composite_standard_deviation",
                "mean_absolute_rank_change_vs_baseline",
                "maximum_absolute_rank_change_vs_baseline",
                "mean_spearman_rank_correlation_vs_baseline",
            ]
        ].to_string(index=False)
    )

    print()
    print("Largest composite-rank disagreements")
    print("-" * 88)
    print(
        disagreements.loc[
            disagreements[
                "representation_dimension"
            ].eq("team_composite")
        ]
        .head(20)[
            [
                "country",
                "specification_id",
                "baseline_rank",
                "within_specification_rank",
                "rank_change_vs_baseline",
                "value_difference_vs_baseline",
            ]
        ]
        .to_string(index=False)
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Study 091A compatibility filter: PASS")
    print("  Complete-XI population contract: PASS")
    print("  Team-dimension row counts: PASS")
    print("  Duplicate representation audit: PASS")
    print("  Finite representation values: PASS")
    print("  Goal-model fitting: NONE")
    print("  Match-outcome usage: NONE")
    print("  Production repository generation: NONE")
    print("  Production configuration change: NONE")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()