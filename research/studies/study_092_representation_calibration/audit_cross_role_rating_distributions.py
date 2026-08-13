# audit_cross_role_rating_distributions

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PLAYER_RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)

ROLE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "role_attribute_manifest.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
)

ROLE_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "role_rating_distribution_summary.csv"
)

ROLE_QUANTILE_PATH = (
    OUTPUT_DIRECTORY
    / "role_rating_quantiles.csv"
)

ROLE_PAIRWISE_PATH = (
    OUTPUT_DIRECTORY
    / "role_rating_pairwise_comparisons.csv"
)

ROLE_EVIDENCE_PATH = (
    OUTPUT_DIRECTORY
    / "role_rating_evidence_diagnostics.csv"
)

ROLE_MANIFEST_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "role_attribute_manifest_summary.csv"
)

PLAYER_ROLE_LONG_PATH = (
    OUTPUT_DIRECTORY
    / "player_role_rating_long.csv"
)

PROVENANCE_PATH = (
    OUTPUT_DIRECTORY
    / "role_scale_provenance.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092a_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_092a_report.md"
)


ROLES = (
    "AM",
    "CB",
    "CM",
    "DM",
    "FB",
    "GK",
    "ST",
    "W",
    "WM",
)

QUANTILES = (
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
)

EVIDENCE_BINS = (
    -np.inf,
    0.25,
    0.50,
    0.75,
    0.90,
    np.inf,
)

EVIDENCE_LABELS = (
    "0.00_to_0.25",
    "0.25_to_0.50",
    "0.50_to_0.75",
    "0.75_to_0.90",
    "0.90_to_1.00",
)


REQUIRED_PLAYER_COLUMNS = {
    "player_id",
    "player",
    "country",
    "evidence_confidence",
    "minutesPlayed",
    "best_role",
    "best_rating",
}

REQUIRED_MANIFEST_COLUMNS = {
    "role",
    "attribute",
    "weight",
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


def finite_numeric(
    values: pd.Series,
    *,
    column_name: str,
    allow_missing: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    if not allow_missing and numeric.isna().any():
        raise ValueError(
            f"{column_name!r} contains missing or non-numeric values."
        )

    finite_values = numeric.dropna().to_numpy(
        dtype=float
    )

    if not np.isfinite(
        finite_values
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


def pooled_standard_deviation(
    values_a: pd.Series,
    values_b: pd.Series,
) -> float:
    variance_a = float(
        values_a.var(ddof=0)
    )

    variance_b = float(
        values_b.var(ddof=0)
    )

    return math.sqrt(
        (
            variance_a
            + variance_b
        ) / 2.0
    )


def standardized_mean_difference(
    values_a: pd.Series,
    values_b: pd.Series,
) -> float | None:
    pooled = pooled_standard_deviation(
        values_a,
        values_b,
    )

    if math.isclose(
        pooled,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None

    return float(
        (
            values_a.mean()
            - values_b.mean()
        ) / pooled
    )


def quantile_overlap_coefficient(
    values_a: pd.Series,
    values_b: pd.Series,
) -> float | None:
    """
    Descriptive overlap of the central 90% intervals.

    1.0 means complete interval overlap.
    0.0 means no overlap.
    """

    lower_a = float(
        values_a.quantile(0.05)
    )
    upper_a = float(
        values_a.quantile(0.95)
    )

    lower_b = float(
        values_b.quantile(0.05)
    )
    upper_b = float(
        values_b.quantile(0.95)
    )

    overlap = max(
        0.0,
        min(
            upper_a,
            upper_b,
        )
        - max(
            lower_a,
            lower_b,
        ),
    )

    union = max(
        upper_a,
        upper_b,
    ) - min(
        lower_a,
        lower_b,
    )

    if math.isclose(
        union,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None

    return float(
        overlap / union
    )


def load_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    player_ratings = pd.read_csv(
        PLAYER_RATINGS_PATH
    )

    role_manifest = pd.read_csv(
        ROLE_MANIFEST_PATH
    )

    require_columns(
        player_ratings,
        REQUIRED_PLAYER_COLUMNS,
        frame_name="Player ratings",
    )

    require_columns(
        role_manifest,
        REQUIRED_MANIFEST_COLUMNS,
        frame_name="Role attribute manifest",
    )

    required_role_columns: set[str] = set()

    for role in ROLES:
        required_role_columns.add(
            f"raw_rating_{role}"
        )
        required_role_columns.add(
            f"rating_{role}"
        )

    require_columns(
        player_ratings,
        required_role_columns,
        frame_name="Player ratings",
    )

    player_ratings = player_ratings.copy()
    role_manifest = role_manifest.copy()

    player_ratings[
        "evidence_confidence"
    ] = finite_numeric(
        player_ratings[
            "evidence_confidence"
        ],
        column_name="evidence_confidence",
        allow_missing=False,
    )

    if (
        player_ratings[
            "evidence_confidence"
        ].lt(0.0).any()
        or player_ratings[
            "evidence_confidence"
        ].gt(1.0).any()
    ):
        raise ValueError(
            "evidence_confidence must lie between zero and one."
        )

    role_manifest[
        "weight"
    ] = finite_numeric(
        role_manifest[
            "weight"
        ],
        column_name="role manifest weight",
        allow_missing=False,
    )

    if role_manifest[
        "weight"
    ].le(0.0).any():
        raise ValueError(
            "Role manifest weights must be positive."
        )

    unknown_roles = sorted(
        set(
            role_manifest[
                "role"
            ].dropna()
        )
        - set(ROLES)
    )

    if unknown_roles:
        raise ValueError(
            "Role manifest contains unexpected roles: "
            f"{unknown_roles}"
        )

    return (
        player_ratings,
        role_manifest,
    )


def build_player_role_long(
    player_ratings: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for row in player_ratings.itertuples(
        index=False
    ):
        evidence_confidence = float(
            row.evidence_confidence
        )

        for role in ROLES:
            raw_value = getattr(
                row,
                f"raw_rating_{role}",
            )

            adjusted_value = getattr(
                row,
                f"rating_{role}",
            )

            if pd.isna(
                raw_value
            ) and pd.isna(
                adjusted_value
            ):
                continue

            if pd.isna(
                raw_value
            ) != pd.isna(
                adjusted_value
            ):
                raise ValueError(
                    "Raw and adjusted role-rating availability "
                    f"do not match for player {row.player_id!r}, "
                    f"role {role!r}."
                )

            raw_value = float(
                raw_value
            )
            adjusted_value = float(
                adjusted_value
            )

            if not math.isfinite(
                raw_value
            ) or not math.isfinite(
                adjusted_value
            ):
                raise ValueError(
                    "Role ratings must be finite when present."
                )

            expected_adjusted = (
                raw_value
                * evidence_confidence
            )

            adjustment_error = abs(
                adjusted_value
                - expected_adjusted
            )

            records.append(
                {
                    "player_id":
                        row.player_id,
                    "player":
                        row.player,
                    "country":
                        row.country,
                    "role":
                        role,
                    "raw_rating":
                        raw_value,
                    "adjusted_rating":
                        adjusted_value,
                    "evidence_confidence":
                        evidence_confidence,
                    "minutesPlayed":
                        row.minutesPlayed,
                    "best_role":
                        row.best_role,
                    "best_rating":
                        row.best_rating,
                    "adjustment_error":
                        adjustment_error,
                    "raw_negative":
                        raw_value < 0.0,
                    "adjusted_negative":
                        adjusted_value < 0.0,
                    "is_best_role":
                        str(
                            row.best_role
                        ) == role,
                }
            )

    output = pd.DataFrame(
        records
    )

    if output.empty:
        raise ValueError(
            "No player-role ratings were available."
        )

    if output[
        "adjustment_error"
    ].max() > 1e-10:
        raise AssertionError(
            "Adjusted role ratings do not reproduce "
            "raw_rating * evidence_confidence."
        )

    output[
        "evidence_band"
    ] = pd.cut(
        output[
            "evidence_confidence"
        ],
        bins=EVIDENCE_BINS,
        labels=EVIDENCE_LABELS,
        right=True,
        include_lowest=True,
    )

    return (
        output
        .sort_values(
            [
                "role",
                "adjusted_rating",
                "player_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def summarize_distribution(
    values: pd.Series,
) -> dict[str, object]:
    return {
        "count":
            len(values),
        "minimum":
            float(
                values.min()
            ),
        "p01":
            float(
                values.quantile(0.01)
            ),
        "p05":
            float(
                values.quantile(0.05)
            ),
        "p25":
            float(
                values.quantile(0.25)
            ),
        "median":
            float(
                values.median()
            ),
        "mean":
            float(
                values.mean()
            ),
        "p75":
            float(
                values.quantile(0.75)
            ),
        "p95":
            float(
                values.quantile(0.95)
            ),
        "p99":
            float(
                values.quantile(0.99)
            ),
        "maximum":
            float(
                values.max()
            ),
        "standard_deviation":
            population_standard_deviation(
                values
            ),
        "median_absolute_deviation":
            median_absolute_deviation(
                values
            ),
        "interquartile_range":
            float(
                values.quantile(0.75)
                - values.quantile(0.25)
            ),
        "p90_range":
            float(
                values.quantile(0.95)
                - values.quantile(0.05)
            ),
        "negative_count":
            int(
                values.lt(0.0).sum()
            ),
        "zero_count":
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
        "positive_count":
            int(
                values.gt(0.0).sum()
            ),
    }


def build_role_distribution_summary(
    player_role_long: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for role, rows in (
        player_role_long.groupby(
            "role",
            sort=True,
        )
    ):
        for scale_name, value_column in (
            (
                "raw",
                "raw_rating",
            ),
            (
                "evidence_adjusted",
                "adjusted_rating",
            ),
        ):
            values = rows[
                value_column
            ]

            records.append(
                {
                    "role":
                        role,
                    "scale":
                        scale_name,
                    "unique_player_count":
                        rows[
                            "player_id"
                        ].nunique(),
                    "country_count":
                        rows[
                            "country"
                        ].nunique(),
                    "mean_evidence_confidence":
                        float(
                            rows[
                                "evidence_confidence"
                            ].mean()
                        ),
                    "median_evidence_confidence":
                        float(
                            rows[
                                "evidence_confidence"
                            ].median()
                        ),
                    **summarize_distribution(
                        values
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_role_quantiles(
    player_role_long: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for role, rows in (
        player_role_long.groupby(
            "role",
            sort=True,
        )
    ):
        for scale_name, value_column in (
            (
                "raw",
                "raw_rating",
            ),
            (
                "evidence_adjusted",
                "adjusted_rating",
            ),
        ):
            values = rows[
                value_column
            ]

            for quantile in QUANTILES:
                records.append(
                    {
                        "role":
                            role,
                        "scale":
                            scale_name,
                        "quantile":
                            quantile,
                        "percentile":
                            quantile
                            * 100.0,
                        "rating_value":
                            float(
                                values.quantile(
                                    quantile
                                )
                            ),
                    }
                )

    return pd.DataFrame(
        records
    )


def build_pairwise_comparisons(
    player_role_long: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for scale_name, value_column in (
        (
            "raw",
            "raw_rating",
        ),
        (
            "evidence_adjusted",
            "adjusted_rating",
        ),
    ):
        role_values = {
            role: rows[
                value_column
            ].reset_index(
                drop=True
            )
            for role, rows
            in player_role_long.groupby(
                "role"
            )
        }

        for left_index, role_a in enumerate(
            ROLES
        ):
            for role_b in ROLES[
                left_index + 1:
            ]:
                values_a = role_values[
                    role_a
                ]
                values_b = role_values[
                    role_b
                ]

                standard_deviation_a = (
                    population_standard_deviation(
                        values_a
                    )
                )

                standard_deviation_b = (
                    population_standard_deviation(
                        values_b
                    )
                )

                if math.isclose(
                    standard_deviation_b,
                    0.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    std_ratio = None
                else:
                    std_ratio = float(
                        standard_deviation_a
                        / standard_deviation_b
                    )

                median_b = float(
                    values_b.median()
                )

                if math.isclose(
                    median_b,
                    0.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    median_ratio = None
                else:
                    median_ratio = float(
                        values_a.median()
                        / median_b
                    )

                records.append(
                    {
                        "scale":
                            scale_name,
                        "role_a":
                            role_a,
                        "role_b":
                            role_b,
                        "count_a":
                            len(values_a),
                        "count_b":
                            len(values_b),
                        "mean_a":
                            float(
                                values_a.mean()
                            ),
                        "mean_b":
                            float(
                                values_b.mean()
                            ),
                        "mean_difference_a_minus_b":
                            float(
                                values_a.mean()
                                - values_b.mean()
                            ),
                        "median_a":
                            float(
                                values_a.median()
                            ),
                        "median_b":
                            median_b,
                        "median_ratio_a_to_b":
                            median_ratio,
                        "standard_deviation_a":
                            standard_deviation_a,
                        "standard_deviation_b":
                            standard_deviation_b,
                        "standard_deviation_ratio_a_to_b":
                            std_ratio,
                        "standardized_mean_difference":
                            standardized_mean_difference(
                                values_a,
                                values_b,
                            ),
                        "central_90_percent_overlap":
                            quantile_overlap_coefficient(
                                values_a,
                                values_b,
                            ),
                    }
                )

    return pd.DataFrame(
        records
    )


def build_evidence_diagnostics(
    player_role_long: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    grouped = player_role_long.groupby(
        [
            "role",
            "evidence_band",
        ],
        observed=True,
        sort=True,
    )

    for (
        role,
        evidence_band,
    ), rows in grouped:
        records.append(
            {
                "role":
                    role,
                "evidence_band":
                    str(
                        evidence_band
                    ),
                "player_role_count":
                    len(rows),
                "mean_evidence_confidence":
                    float(
                        rows[
                            "evidence_confidence"
                        ].mean()
                    ),
                "median_evidence_confidence":
                    float(
                        rows[
                            "evidence_confidence"
                        ].median()
                    ),
                "mean_raw_rating":
                    float(
                        rows[
                            "raw_rating"
                        ].mean()
                    ),
                "mean_adjusted_rating":
                    float(
                        rows[
                            "adjusted_rating"
                        ].mean()
                    ),
                "median_raw_rating":
                    float(
                        rows[
                            "raw_rating"
                        ].median()
                    ),
                "median_adjusted_rating":
                    float(
                        rows[
                            "adjusted_rating"
                        ].median()
                    ),
                "raw_standard_deviation":
                    population_standard_deviation(
                        rows[
                            "raw_rating"
                        ]
                    ),
                "adjusted_standard_deviation":
                    population_standard_deviation(
                        rows[
                            "adjusted_rating"
                        ]
                    ),
                "mean_absolute_shrinkage":
                    float(
                        (
                            rows[
                                "raw_rating"
                            ]
                            - rows[
                                "adjusted_rating"
                            ]
                        )
                        .abs()
                        .mean()
                    ),
                "negative_adjusted_count":
                    int(
                        rows[
                            "adjusted_rating"
                        ].lt(0.0).sum()
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_manifest_summary(
    role_manifest: pd.DataFrame,
) -> pd.DataFrame:
    output = role_manifest.copy()

    role_weight_totals = (
        output.groupby(
            "role"
        )[
            "weight"
        ]
        .transform("sum")
    )

    output[
        "normalized_weight_within_role"
    ] = (
        output[
            "weight"
        ]
        / role_weight_totals
    )

    role_summary = (
        output
        .groupby(
            "role",
            as_index=False,
        )
        .agg(
            attribute_count=(
                "attribute",
                "nunique",
            ),
            total_manifest_weight=(
                "weight",
                "sum",
            ),
            minimum_attribute_weight=(
                "weight",
                "min",
            ),
            maximum_attribute_weight=(
                "weight",
                "max",
            ),
            mean_attribute_weight=(
                "weight",
                "mean",
            ),
            attributes=(
                "attribute",
                lambda values: json.dumps(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                ),
            ),
        )
    )

    return role_summary


def build_provenance_table(
    *,
    player_role_long: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    manifest_summary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    adjusted = distribution_summary.loc[
        distribution_summary[
            "scale"
        ].eq(
            "evidence_adjusted"
        )
    ].set_index(
        "role"
    )

    raw = distribution_summary.loc[
        distribution_summary[
            "scale"
        ].eq(
            "raw"
        )
    ].set_index(
        "role"
    )

    manifest = manifest_summary.set_index(
        "role"
    )

    for role in ROLES:
        raw_mean = float(
            raw.at[
                role,
                "mean",
            ]
        )

        adjusted_mean = float(
            adjusted.at[
                role,
                "mean",
            ]
        )

        if math.isclose(
            raw_mean,
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            mean_retention = None
        else:
            mean_retention = float(
                adjusted_mean
                / raw_mean
            )

        records.append(
            {
                "role":
                    role,
                "manifest_attribute_count":
                    int(
                        manifest.at[
                            role,
                            "attribute_count",
                        ]
                    ),
                "manifest_total_weight":
                    float(
                        manifest.at[
                            role,
                            "total_manifest_weight",
                        ]
                    ),
                "eligible_player_count":
                    int(
                        adjusted.at[
                            role,
                            "count",
                        ]
                    ),
                "raw_mean":
                    raw_mean,
                "raw_median":
                    float(
                        raw.at[
                            role,
                            "median",
                        ]
                    ),
                "raw_standard_deviation":
                    float(
                        raw.at[
                            role,
                            "standard_deviation",
                        ]
                    ),
                "adjusted_mean":
                    adjusted_mean,
                "adjusted_median":
                    float(
                        adjusted.at[
                            role,
                            "median",
                        ]
                    ),
                "adjusted_standard_deviation":
                    float(
                        adjusted.at[
                            role,
                            "standard_deviation",
                        ]
                    ),
                "mean_evidence_confidence":
                    float(
                        adjusted.at[
                            role,
                            "mean_evidence_confidence",
                        ]
                    ),
                "adjusted_to_raw_mean_retention":
                    mean_retention,
                "negative_adjusted_share":
                    float(
                        adjusted.at[
                            role,
                            "negative_count",
                        ]
                        / adjusted.at[
                            role,
                            "count",
                        ]
                    ),
                "cross_role_calibration_applied":
                    False,
                "role_specific_standardization_applied":
                    False,
                "scale_interpretation": (
                    "Comparable within the observed role population; "
                    "cross-role comparability is not guaranteed by "
                    "the current construction pipeline."
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def validate_outputs(
    *,
    player_role_long: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    quantiles: pd.DataFrame,
    pairwise: pd.DataFrame,
    evidence: pd.DataFrame,
    manifest_summary: pd.DataFrame,
    provenance: pd.DataFrame,
) -> None:
    expected_distribution_rows = (
        len(ROLES)
        * 2
    )

    expected_quantile_rows = (
        len(ROLES)
        * 2
        * len(QUANTILES)
    )

    expected_pairwise_rows = (
        2
        * (
            len(ROLES)
            * (
                len(ROLES)
                - 1
            )
            // 2
        )
    )

    if len(
        distribution_summary
    ) != expected_distribution_rows:
        raise AssertionError(
            "Unexpected role-distribution row count."
        )

    if len(
        quantiles
    ) != expected_quantile_rows:
        raise AssertionError(
            "Unexpected role-quantile row count."
        )

    if len(
        pairwise
    ) != expected_pairwise_rows:
        raise AssertionError(
            "Unexpected pairwise role-comparison row count."
        )

    if len(
        manifest_summary
    ) != len(
        ROLES
    ):
        raise AssertionError(
            "Manifest summary does not cover every role."
        )

    if len(
        provenance
    ) != len(
        ROLES
    ):
        raise AssertionError(
            "Provenance output does not cover every role."
        )

    if player_role_long[
        [
            "player_id",
            "role",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Player-role long output contains duplicates."
        )

    if evidence.empty:
        raise AssertionError(
            "Evidence diagnostic output is empty."
        )

    for frame_name, frame in {
        "player-role long":
            player_role_long,
        "distribution summary":
            distribution_summary,
        "quantiles":
            quantiles,
        "pairwise comparisons":
            pairwise,
        "evidence diagnostics":
            evidence,
        "manifest summary":
            manifest_summary,
        "provenance":
            provenance,
    }.items():
        numeric = frame.select_dtypes(
            include="number"
        )

        invalid = numeric.map(
            lambda value: (
                False
                if pd.isna(
                    value
                )
                else not math.isfinite(
                    float(
                        value
                    )
                )
            )
        )

        if invalid.any().any():
            raise AssertionError(
                f"{frame_name} contains non-finite numeric values."
            )


def build_metadata(
    *,
    player_ratings: pd.DataFrame,
    player_role_long: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    evidence: pd.DataFrame,
    manifest_summary: pd.DataFrame,
) -> dict[str, Any]:
    adjusted = distribution_summary.loc[
        distribution_summary[
            "scale"
        ].eq(
            "evidence_adjusted"
        )
    ]

    role_means = adjusted.set_index(
        "role"
    )[
        "mean"
    ]

    goalkeeper_mean = float(
        role_means.loc[
            "GK"
        ]
    )

    outfield_mean = float(
        role_means.drop(
            labels="GK"
        ).mean()
    )

    if math.isclose(
        outfield_mean,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        goalkeeper_to_outfield_mean_ratio = None
    else:
        goalkeeper_to_outfield_mean_ratio = float(
            goalkeeper_mean
            / outfield_mean
        )

    largest_absolute_standardized_difference = float(
        pairwise[
            "standardized_mean_difference"
        ]
        .abs()
        .max()
    )

    goalkeeper_pairwise = pairwise.loc[
        pairwise[
            "role_a"
        ].eq("GK")
        | pairwise[
            "role_b"
        ].eq("GK")
    ]

    smallest_goalkeeper_overlap = float(
        goalkeeper_pairwise[
            "central_90_percent_overlap"
        ]
        .dropna()
        .min()
    )

    return {
        "study_id": "092A",
        "study_name": (
            "Cross-Role Rating Distribution Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "player_record_count":
            len(player_ratings),
        "player_role_record_count":
            len(player_role_long),
        "role_count":
            len(ROLES),
        "distribution_summary_row_count":
            len(distribution_summary),
        "pairwise_comparison_row_count":
            len(pairwise),
        "evidence_diagnostic_row_count":
            len(evidence),
        "manifest_role_count":
            len(manifest_summary),
        "adjusted_goalkeeper_mean":
            goalkeeper_mean,
        "adjusted_outfield_role_mean":
            outfield_mean,
        "goalkeeper_to_outfield_role_mean_ratio":
            goalkeeper_to_outfield_mean_ratio,
        "largest_absolute_standardized_mean_difference":
            largest_absolute_standardized_difference,
        "smallest_goalkeeper_central_90_overlap":
            smallest_goalkeeper_overlap,
        "cross_role_calibration_detected":
            False,
        "calibration_performed":
            False,
        "role_weights_changed":
            False,
        "player_ratings_changed":
            False,
        "team_representations_generated":
            False,
        "goal_model_fitted":
            False,
        "production_runtime_changed":
            False,
        "interpretation_boundary": (
            "This audit describes raw and evidence-adjusted role-rating "
            "distributions. It does not assume that equal numeric values "
            "represent equal football strength across roles."
        ),
        "outputs": [
            ROLE_DISTRIBUTION_PATH.name,
            ROLE_QUANTILE_PATH.name,
            ROLE_PAIRWISE_PATH.name,
            ROLE_EVIDENCE_PATH.name,
            ROLE_MANIFEST_SUMMARY_PATH.name,
            PLAYER_ROLE_LONG_PATH.name,
            PROVENANCE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    distribution_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    evidence: pd.DataFrame,
    manifest_summary: pd.DataFrame,
    provenance: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    adjusted_summary = (
        distribution_summary.loc[
            distribution_summary[
                "scale"
            ].eq(
                "evidence_adjusted"
            )
        ][
            [
                "role",
                "count",
                "mean",
                "median",
                "standard_deviation",
                "p05",
                "p95",
                "negative_count",
                "mean_evidence_confidence",
            ]
        ]
        .sort_values(
            "mean",
            ascending=False,
        )
        .to_dict(
            orient="records"
        )
    )

    raw_summary = (
        distribution_summary.loc[
            distribution_summary[
                "scale"
            ].eq(
                "raw"
            )
        ][
            [
                "role",
                "count",
                "mean",
                "median",
                "standard_deviation",
                "p05",
                "p95",
            ]
        ]
        .sort_values(
            "mean",
            ascending=False,
        )
        .to_dict(
            orient="records"
        )
    )

    goalkeeper_comparisons = (
        pairwise.loc[
            (
                pairwise[
                    "role_a"
                ].eq("GK")
                | pairwise[
                    "role_b"
                ].eq("GK")
            )
            & pairwise[
                "scale"
            ].eq(
                "evidence_adjusted"
            )
        ]
        .sort_values(
            "central_90_percent_overlap"
        )[
            [
                "role_a",
                "role_b",
                "mean_a",
                "mean_b",
                "standardized_mean_difference",
                "central_90_percent_overlap",
                "standard_deviation_ratio_a_to_b",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    evidence_summary = (
        evidence.groupby(
            "role",
            as_index=False,
        )
        .agg(
            evidence_band_count=(
                "evidence_band",
                "nunique",
            ),
            total_player_role_count=(
                "player_role_count",
                "sum",
            ),
            mean_absolute_shrinkage=(
                "mean_absolute_shrinkage",
                "mean",
            ),
        )
        .to_dict(
            orient="records"
        )
    )

    report = f"""# Study 092A — Cross-Role Rating Distribution Audit

## Purpose

Determine whether the current role-rating pipeline produces comparable
numeric scales across football roles before any team-dimension
calibration is attempted.

## Construction provenance

The current player-rating process:

1. selects the attributes assigned to each role;
2. computes a weighted attribute average;
3. writes the result as `raw_rating_<role>`;
4. multiplies it by `evidence_confidence`;
5. writes the result as `rating_<role>`.

No explicit cross-role calibration is applied after those calculations.

## Coverage

- Player records: {metadata["player_record_count"]}
- Player-role observations: {metadata["player_role_record_count"]}
- Roles: {metadata["role_count"]}
- Pairwise role comparisons:
  {metadata["pairwise_comparison_row_count"]}

## Raw role-rating distributions

{json.dumps(
    raw_summary,
    indent=2,
    default=str,
)}

## Evidence-adjusted role-rating distributions

{json.dumps(
    adjusted_summary,
    indent=2,
    default=str,
)}

## Goalkeeper comparisons

{json.dumps(
    goalkeeper_comparisons,
    indent=2,
    default=str,
)}

## Aggregate scale diagnostics

- Adjusted goalkeeper mean:
  {metadata["adjusted_goalkeeper_mean"]}
- Mean adjusted outfield-role mean:
  {metadata["adjusted_outfield_role_mean"]}
- Goalkeeper-to-outfield role-mean ratio:
  {metadata["goalkeeper_to_outfield_role_mean_ratio"]}
- Largest absolute standardized role-mean difference:
  {metadata["largest_absolute_standardized_mean_difference"]}
- Smallest goalkeeper central-90% overlap:
  {metadata["smallest_goalkeeper_central_90_overlap"]}

## Role-manifest summary

{json.dumps(
    manifest_summary.to_dict(
        orient="records"
    ),
    indent=2,
    default=str,
)}

## Evidence adjustment summary

{json.dumps(
    evidence_summary,
    indent=2,
    default=str,
)}

## Provenance interpretation

The audit distinguishes two possibilities:

1. goalkeeper and outfield scales already diverge in raw role ratings;
2. comparable raw scales diverge only after evidence adjustment.

The generated raw and adjusted summaries should be compared before any
calibration method is proposed.

## Methodological boundary

This study:

- changes no player rating;
- changes no role attribute;
- changes no manifest weight;
- changes no role projection;
- performs no normalization;
- creates no team repository;
- fits no goal model;
- makes no claim that equal values are cross-role equivalent.

## Result

**OVERALL RESULT: {metadata["status"]}**

The current cross-role rating scales were audited without performing
calibration.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 092A — CROSS-ROLE RATING DISTRIBUTION AUDIT"
    )
    print("=" * 88)

    (
        player_ratings,
        role_manifest,
    ) = load_inputs()

    player_role_long = (
        build_player_role_long(
            player_ratings
        )
    )

    distribution_summary = (
        build_role_distribution_summary(
            player_role_long
        )
    )

    quantiles = build_role_quantiles(
        player_role_long
    )

    pairwise = build_pairwise_comparisons(
        player_role_long
    )

    evidence = build_evidence_diagnostics(
        player_role_long
    )

    manifest_summary = (
        build_manifest_summary(
            role_manifest
        )
    )

    provenance = build_provenance_table(
        player_role_long=(
            player_role_long
        ),
        distribution_summary=(
            distribution_summary
        ),
        manifest_summary=manifest_summary,
    )

    validate_outputs(
        player_role_long=player_role_long,
        distribution_summary=(
            distribution_summary
        ),
        quantiles=quantiles,
        pairwise=pairwise,
        evidence=evidence,
        manifest_summary=manifest_summary,
        provenance=provenance,
    )

    metadata = build_metadata(
        player_ratings=player_ratings,
        player_role_long=player_role_long,
        distribution_summary=(
            distribution_summary
        ),
        pairwise=pairwise,
        evidence=evidence,
        manifest_summary=manifest_summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_role_long.to_csv(
        PLAYER_ROLE_LONG_PATH,
        index=False,
    )

    distribution_summary.to_csv(
        ROLE_DISTRIBUTION_PATH,
        index=False,
    )

    quantiles.to_csv(
        ROLE_QUANTILE_PATH,
        index=False,
    )

    pairwise.to_csv(
        ROLE_PAIRWISE_PATH,
        index=False,
    )

    evidence.to_csv(
        ROLE_EVIDENCE_PATH,
        index=False,
    )

    manifest_summary.to_csv(
        ROLE_MANIFEST_SUMMARY_PATH,
        index=False,
    )

    provenance.to_csv(
        PROVENANCE_PATH,
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
        distribution_summary=(
            distribution_summary
        ),
        pairwise=pairwise,
        evidence=evidence,
        manifest_summary=manifest_summary,
        provenance=provenance,
        metadata=metadata,
    )

    print()
    print("Evidence-adjusted role distributions")
    print("-" * 88)
    print(
        distribution_summary.loc[
            distribution_summary[
                "scale"
            ].eq(
                "evidence_adjusted"
            )
        ][
            [
                "role",
                "count",
                "mean",
                "median",
                "standard_deviation",
                "p05",
                "p95",
                "negative_count",
            ]
        ]
        .sort_values(
            "mean",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Cross-role scale summary")
    print("-" * 88)
    print(
        "  Goalkeeper-to-outfield role-mean ratio: "
        f"{metadata['goalkeeper_to_outfield_role_mean_ratio']}"
    )
    print(
        "  Largest absolute standardized mean difference: "
        f"{metadata['largest_absolute_standardized_mean_difference']}"
    )
    print(
        "  Smallest goalkeeper central-90% overlap: "
        f"{metadata['smallest_goalkeeper_central_90_overlap']}"
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Player-rating schema: PASS")
    print("  Role-manifest schema: PASS")
    print("  Raw-to-adjusted identity: PASS")
    print("  Role distribution coverage: PASS")
    print("  Pairwise comparison coverage: PASS")
    print("  Calibration performed: NO")
    print("  Player ratings changed: NO")
    print("  Team repository generated: NO")
    print("  Production behavior changed: NO")

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