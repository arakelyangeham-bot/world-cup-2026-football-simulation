# audit_dimension_scale_compatibility.py

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


WIDE_REPRESENTATION_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_team_representations_wide.csv"
)

LONG_REPRESENTATION_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_team_representations_long.csv"
)

PLAYER_POPULATION_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_player_population.csv"
)


DIMENSION_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "dimension_scale_distributions.csv"
)

DIMENSION_PAIRWISE_PATH = (
    OUTPUT_DIRECTORY
    / "dimension_scale_pairwise_comparisons.csv"
)

OUTFIELD_RANKING_PATH = (
    OUTPUT_DIRECTORY
    / "outfield_only_team_rankings.csv"
)

STANDARDIZED_COMPOSITE_PATH = (
    OUTPUT_DIRECTORY
    / "standardized_dimension_composites.csv"
)

PERCENTILE_COMPOSITE_PATH = (
    OUTPUT_DIRECTORY
    / "percentile_dimension_composites.csv"
)

COMPOSITE_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "composite_rank_comparisons.csv"
)

GOALKEEPER_INFLUENCE_PATH = (
    OUTPUT_DIRECTORY
    / "goalkeeper_composite_influence.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_091b1_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_091b1_report.md"
)


DIMENSIONS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
)

OUTFIELD_DIMENSIONS = (
    "attack",
    "midfield",
    "defense",
)

COMPOSITE_WEIGHTS = {
    "attack": 0.35,
    "midfield": 0.25,
    "defense": 0.25,
    "goalkeeper": 0.15,
}


REQUIRED_WIDE_COLUMNS = {
    "country",
    "specification_id",
    "aggregation_family",
    "display_name",
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "outfield_mean",
    "team_composite",
    "team_composite_rank",
}

REQUIRED_PLAYER_COLUMNS = {
    "country",
    "player_id",
    "projected_attack",
    "projected_midfield",
    "projected_defense",
    "projected_goalkeeper",
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


def finite_numeric_series(
    values: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    if numeric.isna().any():
        raise ValueError(
            f"{column_name!r} contains missing or non-numeric values."
        )

    if not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{column_name!r} contains non-finite values."
        )

    return numeric


def population_standard_deviation(
    values: pd.Series,
) -> float:
    return float(
        values.std(ddof=0)
    )


def median_absolute_deviation(
    values: pd.Series,
) -> float:
    median = float(
        values.median()
    )

    return float(
        (
            values
            - median
        )
        .abs()
        .median()
    )


def quantile_range(
    values: pd.Series,
    *,
    lower: float,
    upper: float,
) -> float:
    return float(
        values.quantile(upper)
        - values.quantile(lower)
    )


def load_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    wide = pd.read_csv(
        WIDE_REPRESENTATION_PATH
    )

    long = pd.read_csv(
        LONG_REPRESENTATION_PATH
    )

    player_population = pd.read_csv(
        PLAYER_POPULATION_PATH
    )

    require_columns(
        wide,
        REQUIRED_WIDE_COLUMNS,
        frame_name="Wide team representations",
    )

    require_columns(
        player_population,
        REQUIRED_PLAYER_COLUMNS,
        frame_name="Expected-lineup player population",
    )

    wide = wide.copy()
    player_population = player_population.copy()

    for dimension in DIMENSIONS:
        wide[dimension] = finite_numeric_series(
            wide[dimension],
            column_name=dimension,
        )

    for column in (
        "projected_attack",
        "projected_midfield",
        "projected_defense",
        "projected_goalkeeper",
    ):
        player_population[column] = (
            finite_numeric_series(
                player_population[column],
                column_name=column,
            )
        )

    return (
        wide,
        long,
        player_population,
    )


def build_dimension_distributions(
    wide: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for specification_id, rows in wide.groupby(
        "specification_id",
        sort=True,
    ):
        metadata = rows.iloc[0]

        for dimension in DIMENSIONS:
            values = rows[dimension]

            records.append(
                {
                    "specification_id":
                        specification_id,
                    "aggregation_family":
                        metadata[
                            "aggregation_family"
                        ],
                    "display_name":
                        metadata[
                            "display_name"
                        ],
                    "dimension":
                        dimension,
                    "team_count":
                        len(values),
                    "minimum":
                        float(values.min()),
                    "p05":
                        float(values.quantile(0.05)),
                    "p25":
                        float(values.quantile(0.25)),
                    "median":
                        float(values.median()),
                    "mean":
                        float(values.mean()),
                    "p75":
                        float(values.quantile(0.75)),
                    "p95":
                        float(values.quantile(0.95)),
                    "maximum":
                        float(values.max()),
                    "standard_deviation":
                        population_standard_deviation(
                            values
                        ),
                    "median_absolute_deviation":
                        median_absolute_deviation(
                            values
                        ),
                    "interquartile_range":
                        quantile_range(
                            values,
                            lower=0.25,
                            upper=0.75,
                        ),
                    "p90_range":
                        quantile_range(
                            values,
                            lower=0.05,
                            upper=0.95,
                        ),
                    "negative_value_count":
                        int(
                            values.lt(0.0).sum()
                        ),
                    "zero_value_count":
                        int(
                            np.isclose(
                                values.to_numpy(
                                    dtype=float
                                ),
                                0.0,
                                rtol=1e-12,
                                atol=1e-12,
                            ).sum()
                        ),
                }
            )

    return pd.DataFrame(records)


def build_pairwise_scale_comparisons(
    distributions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    metrics = (
        "mean",
        "median",
        "standard_deviation",
        "interquartile_range",
        "p90_range",
    )

    for specification_id, rows in (
        distributions.groupby(
            "specification_id",
            sort=True,
        )
    ):
        by_dimension = (
            rows
            .set_index("dimension")
        )

        for left_index, dimension_a in enumerate(
            DIMENSIONS
        ):
            for dimension_b in DIMENSIONS[
                left_index + 1:
            ]:
                record: dict[str, object] = {
                    "specification_id":
                        specification_id,
                    "dimension_a":
                        dimension_a,
                    "dimension_b":
                        dimension_b,
                }

                for metric in metrics:
                    value_a = float(
                        by_dimension.at[
                            dimension_a,
                            metric,
                        ]
                    )

                    value_b = float(
                        by_dimension.at[
                            dimension_b,
                            metric,
                        ]
                    )

                    if math.isclose(
                        value_b,
                        0.0,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        ratio = None
                    else:
                        ratio = float(
                            value_a / value_b
                        )

                    record[
                        f"{metric}_a"
                    ] = value_a

                    record[
                        f"{metric}_b"
                    ] = value_b

                    record[
                        f"{metric}_ratio_a_to_b"
                    ] = ratio

                    record[
                        f"{metric}_absolute_difference"
                    ] = abs(
                        value_a
                        - value_b
                    )

                records.append(record)

    return pd.DataFrame(records)


def add_outfield_only_rankings(
    wide: pd.DataFrame,
) -> pd.DataFrame:
    output = wide.copy()

    output[
        "outfield_weighted_composite"
    ] = (
        output["attack"]
        * (
            COMPOSITE_WEIGHTS[
                "attack"
            ]
            / 0.85
        )
        + output["midfield"]
        * (
            COMPOSITE_WEIGHTS[
                "midfield"
            ]
            / 0.85
        )
        + output["defense"]
        * (
            COMPOSITE_WEIGHTS[
                "defense"
            ]
            / 0.85
        )
    )

    output[
        "outfield_weighted_rank"
    ] = (
        output
        .groupby(
            "specification_id"
        )[
            "outfield_weighted_composite"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    output[
        "outfield_mean_rank"
    ] = (
        output
        .groupby(
            "specification_id"
        )[
            "outfield_mean"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return output


def standardize_within_specification(
    wide: pd.DataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for specification_id, rows in wide.groupby(
        "specification_id",
        sort=True,
    ):
        output = rows.copy()

        for dimension in DIMENSIONS:
            mean = float(
                output[
                    dimension
                ].mean()
            )

            standard_deviation = float(
                output[
                    dimension
                ].std(ddof=0)
            )

            standardized_column = (
                f"{dimension}_z"
            )

            if math.isclose(
                standard_deviation,
                0.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                output[
                    standardized_column
                ] = 0.0

            else:
                output[
                    standardized_column
                ] = (
                    output[
                        dimension
                    ]
                    - mean
                ) / standard_deviation

        output[
            "standardized_composite"
        ] = sum(
            output[
                f"{dimension}_z"
            ]
            * weight
            for dimension, weight
            in COMPOSITE_WEIGHTS.items()
        )

        output[
            "standardized_composite_rank"
        ] = output[
            "standardized_composite"
        ].rank(
            method="min",
            ascending=False,
        ).astype(int)

        records.append(output)

    return (
        pd.concat(
            records,
            ignore_index=True,
        )
        .sort_values(
            [
                "specification_id",
                "standardized_composite_rank",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def percentile_normalize_within_specification(
    wide: pd.DataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for specification_id, rows in wide.groupby(
        "specification_id",
        sort=True,
    ):
        output = rows.copy()

        team_count = len(
            output
        )

        for dimension in DIMENSIONS:
            percentile_column = (
                f"{dimension}_percentile"
            )

            output[
                percentile_column
            ] = (
                output[
                    dimension
                ]
                .rank(
                    method="average",
                    ascending=True,
                )
                - 1.0
            ) / max(
                team_count - 1,
                1,
            )

        output[
            "percentile_composite"
        ] = sum(
            output[
                f"{dimension}_percentile"
            ]
            * weight
            for dimension, weight
            in COMPOSITE_WEIGHTS.items()
        )

        output[
            "percentile_composite_rank"
        ] = output[
            "percentile_composite"
        ].rank(
            method="min",
            ascending=False,
        ).astype(int)

        records.append(output)

    return (
        pd.concat(
            records,
            ignore_index=True,
        )
        .sort_values(
            [
                "specification_id",
                "percentile_composite_rank",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def build_goalkeeper_influence(
    outfield_rankings: pd.DataFrame,
) -> pd.DataFrame:
    output = outfield_rankings.copy()

    output[
        "goalkeeper_weighted_contribution"
    ] = (
        output[
            "goalkeeper"
        ]
        * COMPOSITE_WEIGHTS[
            "goalkeeper"
        ]
    )

    output[
        "attack_weighted_contribution"
    ] = (
        output[
            "attack"
        ]
        * COMPOSITE_WEIGHTS[
            "attack"
        ]
    )

    output[
        "midfield_weighted_contribution"
    ] = (
        output[
            "midfield"
        ]
        * COMPOSITE_WEIGHTS[
            "midfield"
        ]
    )

    output[
        "defense_weighted_contribution"
    ] = (
        output[
            "defense"
        ]
        * COMPOSITE_WEIGHTS[
            "defense"
        ]
    )

    output[
        "outfield_weighted_contribution"
    ] = (
        output[
            "attack_weighted_contribution"
        ]
        + output[
            "midfield_weighted_contribution"
        ]
        + output[
            "defense_weighted_contribution"
        ]
    )

    output[
        "goalkeeper_share_of_raw_composite"
    ] = np.where(
        output[
            "team_composite"
        ].abs().gt(1e-12),
        (
            output[
                "goalkeeper_weighted_contribution"
            ]
            / output[
                "team_composite"
            ]
        ),
        np.nan,
    )

    output[
        "raw_vs_outfield_rank_change"
    ] = (
        output[
            "outfield_weighted_rank"
        ]
        - output[
            "team_composite_rank"
        ]
    )

    output[
        "absolute_raw_vs_outfield_rank_change"
    ] = output[
        "raw_vs_outfield_rank_change"
    ].abs()

    output[
        "goalkeeper_contribution_exceeds_outfield"
    ] = (
        output[
            "goalkeeper_weighted_contribution"
        ]
        > output[
            "outfield_weighted_contribution"
        ]
    )

    return (
        output
        .sort_values(
            [
                "absolute_raw_vs_outfield_rank_change",
                "goalkeeper_share_of_raw_composite",
                "country",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_composite_rank_comparisons(
    *,
    outfield_rankings: pd.DataFrame,
    standardized: pd.DataFrame,
    percentile: pd.DataFrame,
) -> pd.DataFrame:
    comparison = outfield_rankings[
        [
            "country",
            "specification_id",
            "team_composite",
            "team_composite_rank",
            "outfield_mean",
            "outfield_mean_rank",
            "outfield_weighted_composite",
            "outfield_weighted_rank",
        ]
    ].merge(
        standardized[
            [
                "country",
                "specification_id",
                "standardized_composite",
                "standardized_composite_rank",
            ]
        ],
        on=[
            "country",
            "specification_id",
        ],
        how="inner",
        validate="one_to_one",
    ).merge(
        percentile[
            [
                "country",
                "specification_id",
                "percentile_composite",
                "percentile_composite_rank",
            ]
        ],
        on=[
            "country",
            "specification_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    for rank_column in (
        "outfield_mean_rank",
        "outfield_weighted_rank",
        "standardized_composite_rank",
        "percentile_composite_rank",
    ):
        difference_column = (
            f"{rank_column}_minus_raw_rank"
        )

        comparison[
            difference_column
        ] = (
            comparison[
                rank_column
            ]
            - comparison[
                "team_composite_rank"
            ]
        )

        comparison[
            f"absolute_{difference_column}"
        ] = comparison[
            difference_column
        ].abs()

    return (
        comparison
        .sort_values(
            [
                "specification_id",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def validate_outputs(
    *,
    wide: pd.DataFrame,
    distributions: pd.DataFrame,
    pairwise: pd.DataFrame,
    outfield_rankings: pd.DataFrame,
    standardized: pd.DataFrame,
    percentile: pd.DataFrame,
    comparisons: pd.DataFrame,
    goalkeeper_influence: pd.DataFrame,
) -> None:
    specification_count = (
        wide[
            "specification_id"
        ].nunique()
    )

    expected_distribution_rows = (
        specification_count
        * len(DIMENSIONS)
    )

    expected_pairwise_rows = (
        specification_count
        * 6
    )

    if len(
        distributions
    ) != expected_distribution_rows:
        raise AssertionError(
            "Unexpected dimension-distribution row count."
        )

    if len(
        pairwise
    ) != expected_pairwise_rows:
        raise AssertionError(
            "Unexpected pairwise scale-comparison row count."
        )

    for frame_name, frame in {
        "outfield rankings":
            outfield_rankings,
        "standardized composites":
            standardized,
        "percentile composites":
            percentile,
        "composite comparisons":
            comparisons,
        "goalkeeper influence":
            goalkeeper_influence,
    }.items():
        if len(frame) != len(wide):
            raise AssertionError(
                f"{frame_name} does not match the wide "
                "representation row count."
            )

    numeric_frames = {
        "distributions":
            distributions,
        "outfield rankings":
            outfield_rankings,
        "standardized composites":
            standardized,
        "percentile composites":
            percentile,
    }

    for frame_name, frame in (
        numeric_frames.items()
    ):
        numeric = frame.select_dtypes(
            include="number"
        )

        invalid = numeric.map(
            lambda value: (
                False
                if pd.isna(value)
                else not math.isfinite(
                    float(value)
                )
            )
        )

        if invalid.any().any():
            raise AssertionError(
                f"{frame_name} contains non-finite values."
            )


def build_metadata(
    *,
    wide: pd.DataFrame,
    distributions: pd.DataFrame,
    pairwise: pd.DataFrame,
    comparisons: pd.DataFrame,
    goalkeeper_influence: pd.DataFrame,
) -> dict[str, Any]:
    baseline_rows = wide.loc[
        wide[
            "specification_id"
        ].eq("top5_arithmetic")
    ]

    goalkeeper_mean = float(
        baseline_rows[
            "goalkeeper"
        ].mean()
    )

    outfield_dimension_mean = float(
        baseline_rows[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ]
        .to_numpy(
            dtype=float
        )
        .mean()
    )

    if math.isclose(
        outfield_dimension_mean,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        goalkeeper_to_outfield_mean_ratio = None
    else:
        goalkeeper_to_outfield_mean_ratio = (
            goalkeeper_mean
            / outfield_dimension_mean
        )

    teams_gk_dominant = int(
        goalkeeper_influence[
            "goalkeeper_contribution_exceeds_outfield"
        ].sum()
    )

    maximum_raw_to_standardized_rank_change = int(
        comparisons[
            "absolute_standardized_composite_rank_minus_raw_rank"
        ].max()
    )

    maximum_raw_to_outfield_rank_change = int(
        comparisons[
            "absolute_outfield_weighted_rank_minus_raw_rank"
        ].max()
    )

    return {
        "study_id": "091B1",
        "study_name": (
            "Dimension Scale Compatibility Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "team_specification_row_count":
            len(wide),
        "team_count":
            wide[
                "country"
            ].nunique(),
        "aggregation_specification_count":
            wide[
                "specification_id"
            ].nunique(),
        "dimension_distribution_row_count":
            len(distributions),
        "pairwise_scale_comparison_row_count":
            len(pairwise),
        "baseline_goalkeeper_mean":
            goalkeeper_mean,
        "baseline_outfield_dimension_mean":
            outfield_dimension_mean,
        "goalkeeper_to_outfield_mean_ratio":
            goalkeeper_to_outfield_mean_ratio,
        "goalkeeper_dominant_team_specification_rows":
            teams_gk_dominant,
        "maximum_raw_to_standardized_rank_change":
            maximum_raw_to_standardized_rank_change,
        "maximum_raw_to_outfield_rank_change":
            maximum_raw_to_outfield_rank_change,
        "raw_team_composite_validated":
            False,
        "standardized_composite_production_candidate":
            False,
        "percentile_composite_production_candidate":
            False,
        "outfield_only_production_candidate":
            False,
        "goal_model_fitted":
            False,
        "production_repository_changed":
            False,
        "production_runtime_changed":
            False,
        "interpretation_boundary": (
            "Standardized, percentile, and outfield-only composites "
            "are diagnostic alternatives used to expose dimensional "
            "scale effects. None is promoted as a production formula."
        ),
        "outputs": [
            DIMENSION_DISTRIBUTION_PATH.name,
            DIMENSION_PAIRWISE_PATH.name,
            OUTFIELD_RANKING_PATH.name,
            STANDARDIZED_COMPOSITE_PATH.name,
            PERCENTILE_COMPOSITE_PATH.name,
            COMPOSITE_COMPARISON_PATH.name,
            GOALKEEPER_INFLUENCE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    distributions: pd.DataFrame,
    comparisons: pd.DataFrame,
    goalkeeper_influence: pd.DataFrame,
    standardized: pd.DataFrame,
    percentile: pd.DataFrame,
    outfield_rankings: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    baseline_distributions = (
        distributions.loc[
            distributions[
                "specification_id"
            ].eq("top5_arithmetic")
        ][
            [
                "dimension",
                "mean",
                "median",
                "standard_deviation",
                "minimum",
                "maximum",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    largest_goalkeeper_rank_effects = (
        goalkeeper_influence
        .head(20)[
            [
                "country",
                "specification_id",
                "team_composite_rank",
                "outfield_weighted_rank",
                "raw_vs_outfield_rank_change",
                "goalkeeper",
                "goalkeeper_weighted_contribution",
                "outfield_weighted_contribution",
                "goalkeeper_share_of_raw_composite",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    top_teams: dict[
        str,
        dict[
            str,
            list[str],
        ],
    ] = {}

    for specification_id in sorted(
        outfield_rankings[
            "specification_id"
        ].unique()
    ):
        top_teams[
            specification_id
        ] = {
            "raw_composite": (
                outfield_rankings.loc[
                    outfield_rankings[
                        "specification_id"
                    ].eq(specification_id)
                ]
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
            ),
            "outfield_only": (
                outfield_rankings.loc[
                    outfield_rankings[
                        "specification_id"
                    ].eq(specification_id)
                ]
                .sort_values(
                    [
                        "outfield_weighted_rank",
                        "country",
                    ]
                )
                .head(10)[
                    "country"
                ]
                .tolist()
            ),
            "standardized": (
                standardized.loc[
                    standardized[
                        "specification_id"
                    ].eq(specification_id)
                ]
                .sort_values(
                    [
                        "standardized_composite_rank",
                        "country",
                    ]
                )
                .head(10)[
                    "country"
                ]
                .tolist()
            ),
            "percentile": (
                percentile.loc[
                    percentile[
                        "specification_id"
                    ].eq(specification_id)
                ]
                .sort_values(
                    [
                        "percentile_composite_rank",
                        "country",
                    ]
                )
                .head(10)[
                    "country"
                ]
                .tolist()
            ),
        }

    largest_normalization_rank_changes = (
        comparisons
        .sort_values(
            "absolute_standardized_composite_rank_minus_raw_rank",
            ascending=False,
        )
        .head(20)[
            [
                "country",
                "specification_id",
                "team_composite_rank",
                "standardized_composite_rank",
                "standardized_composite_rank_minus_raw_rank",
                "outfield_weighted_rank",
                "percentile_composite_rank",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    report = f"""# Study 091B1 — Dimension Scale Compatibility Audit

## Purpose

Audit whether attack, midfield, defense, and goalkeeper values can be
combined directly using the descriptive Study 091B team-composite
weights.

## Scope

- Teams: {metadata["team_count"]}
- Aggregation specifications:
  {metadata["aggregation_specification_count"]}
- Team-specification rows:
  {metadata["team_specification_row_count"]}

## Baseline dimension distributions

{json.dumps(
    baseline_distributions,
    indent=2,
)}

## Primary scale finding

- Baseline goalkeeper mean:
  {metadata["baseline_goalkeeper_mean"]}
- Baseline mean across outfield dimensions:
  {metadata["baseline_outfield_dimension_mean"]}
- Goalkeeper-to-outfield mean ratio:
  {metadata["goalkeeper_to_outfield_mean_ratio"]}
- Team-specification rows where weighted goalkeeper contribution exceeds
  the combined weighted outfield contribution:
  {metadata["goalkeeper_dominant_team_specification_rows"]}

These figures test whether the nominal 15% goalkeeper weight behaves as
a true 15% contribution on the current numerical scales.

## Largest goalkeeper-driven rank effects

{json.dumps(
    largest_goalkeeper_rank_effects,
    indent=2,
    default=str,
)}

## Top-team comparison by diagnostic composite

{json.dumps(
    top_teams,
    indent=2,
)}

## Largest raw-to-standardized rank changes

{json.dumps(
    largest_normalization_rank_changes,
    indent=2,
    default=str,
)}

## Diagnostic alternatives

This audit constructs three non-production alternatives:

1. outfield-only weighted composite;
2. within-specification z-score composite;
3. within-specification percentile composite.

These alternatives are not proposed as production formulas. They are
diagnostic tools used to identify whether rankings are driven by genuine
cross-dimensional strength or incompatible raw scales.

## Interpretation

The original Study 091B raw team composite should remain provisional
unless the dimensional distributions support direct addition.

Within-dimension aggregation results remain valid regardless of the
composite-scale finding.

## Methodological boundary

This study:

- changes no player rating;
- changes no aggregation formula;
- changes no role projection;
- fits no goal model;
- generates no production repository;
- selects no corrected composite;
- changes no production runtime.

## Result

**OVERALL RESULT: {metadata["status"]}**

The dimensional scale audit completed successfully. The validity of the
raw Study 091B composite must be interpreted from the generated scale and
rank diagnostics rather than assumed from the nominal weights.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 091B1 — DIMENSION SCALE COMPATIBILITY AUDIT"
    )
    print("=" * 88)

    (
        wide,
        _long,
        _player_population,
    ) = load_inputs()

    distributions = (
        build_dimension_distributions(
            wide
        )
    )

    pairwise = (
        build_pairwise_scale_comparisons(
            distributions
        )
    )

    outfield_rankings = (
        add_outfield_only_rankings(
            wide
        )
    )

    standardized = (
        standardize_within_specification(
            wide
        )
    )

    percentile = (
        percentile_normalize_within_specification(
            wide
        )
    )

    comparisons = (
        build_composite_rank_comparisons(
            outfield_rankings=(
                outfield_rankings
            ),
            standardized=standardized,
            percentile=percentile,
        )
    )

    goalkeeper_influence = (
        build_goalkeeper_influence(
            outfield_rankings
        )
    )

    validate_outputs(
        wide=wide,
        distributions=distributions,
        pairwise=pairwise,
        outfield_rankings=(
            outfield_rankings
        ),
        standardized=standardized,
        percentile=percentile,
        comparisons=comparisons,
        goalkeeper_influence=(
            goalkeeper_influence
        ),
    )

    metadata = build_metadata(
        wide=wide,
        distributions=distributions,
        pairwise=pairwise,
        comparisons=comparisons,
        goalkeeper_influence=(
            goalkeeper_influence
        ),
    )

    distributions.to_csv(
        DIMENSION_DISTRIBUTION_PATH,
        index=False,
    )

    pairwise.to_csv(
        DIMENSION_PAIRWISE_PATH,
        index=False,
    )

    outfield_rankings.to_csv(
        OUTFIELD_RANKING_PATH,
        index=False,
    )

    standardized.to_csv(
        STANDARDIZED_COMPOSITE_PATH,
        index=False,
    )

    percentile.to_csv(
        PERCENTILE_COMPOSITE_PATH,
        index=False,
    )

    comparisons.to_csv(
        COMPOSITE_COMPARISON_PATH,
        index=False,
    )

    goalkeeper_influence.to_csv(
        GOALKEEPER_INFLUENCE_PATH,
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
        distributions=distributions,
        comparisons=comparisons,
        goalkeeper_influence=(
            goalkeeper_influence
        ),
        standardized=standardized,
        percentile=percentile,
        outfield_rankings=(
            outfield_rankings
        ),
        metadata=metadata,
    )

    print()
    print("Baseline dimension distributions")
    print("-" * 88)
    print(
        distributions.loc[
            distributions[
                "specification_id"
            ].eq("top5_arithmetic")
        ][
            [
                "dimension",
                "mean",
                "median",
                "standard_deviation",
                "minimum",
                "maximum",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Scale summary")
    print("-" * 88)
    print(
        "  Goalkeeper-to-outfield mean ratio: "
        f"{metadata['goalkeeper_to_outfield_mean_ratio']}"
    )
    print(
        "  GK-dominant team-specification rows: "
        f"{metadata['goalkeeper_dominant_team_specification_rows']}"
    )
    print(
        "  Maximum raw-to-standardized rank change: "
        f"{metadata['maximum_raw_to_standardized_rank_change']}"
    )
    print(
        "  Maximum raw-to-outfield rank change: "
        f"{metadata['maximum_raw_to_outfield_rank_change']}"
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Dimension distributions: PASS")
    print("  Pairwise scale audit: PASS")
    print("  Outfield-only diagnostic: PASS")
    print("  Standardized composite diagnostic: PASS")
    print("  Percentile composite diagnostic: PASS")
    print("  Production composite replacement: NONE")
    print("  Goal-model fitting: NONE")
    print("  Production repository change: NONE")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()