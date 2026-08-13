# compare_feature_transformations.py

from __future__ import annotations

import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_model_features.csv"
)

COMPETITION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_manifest.csv"
)

COMPETITION_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_feature_manifest.csv"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_registry.csv"
)

FEATURE_ATTRIBUTE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

ROLE_ATTRIBUTE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "role_attribute_manifest.csv"
)

EXPECTED_LINEUPS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expected_lineups.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "feature_transformation_audit"
)

FEATURE_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "feature_transformation_distributions.csv"
)

ATTRIBUTE_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "attribute_transformation_distributions.csv"
)

ROLE_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "role_transformation_distributions.csv"
)

ROLE_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "role_scale_compatibility_by_transformation.csv"
)

PLAYER_ATTRIBUTE_PATH = (
    OUTPUT_DIRECTORY
    / "player_attributes_by_transformation.csv"
)

PLAYER_ROLE_PATH = (
    OUTPUT_DIRECTORY
    / "player_roles_by_transformation.csv"
)

EXPECTED_XI_DIMENSION_PATH = (
    OUTPUT_DIRECTORY
    / "expected_xi_dimensions_by_transformation.csv"
)

TEAM_SCALE_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "expected_xi_dimension_scale_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_092b_report.md"
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

TRANSFORMATIONS = (
    "global_zscore",
    "winsorized_zscore",
    "robust_zscore",
    "percentile_normal",
)

WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

ROBUST_SCALE_CONSTANT = 1.4826

EXPECTED_XI_REQUIRED_PLAYERS = 11


PROJECTION_WEIGHTS = {
    "attack": {
        "ST": 0.40,
        "W": 0.25,
        "AM": 0.20,
        "CM": 0.10,
        "FB": 0.05,
    },
    "midfield": {
        "CM": 0.35,
        "DM": 0.25,
        "AM": 0.20,
        "WM": 0.10,
        "FB": 0.10,
    },
    "defense": {
        "CB": 0.40,
        "FB": 0.25,
        "DM": 0.25,
        "GK": 0.10,
    },
    "goalkeeper": {
        "GK": 1.00,
    },
}


REQUIRED_FEATURE_COLUMNS = {
    "player_id",
    "player",
    "competition_id",
    "season_id",
    "season_year",
    "competition",
    "minutesPlayed",
}

REQUIRED_COMPETITION_COLUMNS = {
    "competition_id",
    "season_id",
    "recency_weight",
    "competition_importance",
}

REQUIRED_COMPETITION_FEATURE_COLUMNS = {
    "competition",
    "season_year",
    "feature",
    "available",
}

REQUIRED_REGISTRY_COLUMNS = {
    "player_id",
    "canonical_player_id",
    "eligible_roles",
    "country",
}

REQUIRED_FEATURE_MANIFEST_COLUMNS = {
    "feature",
    "attribute",
    "weight",
}

REQUIRED_ROLE_MANIFEST_COLUMNS = {
    "role",
    "attribute",
    "weight",
}

REQUIRED_LINEUP_COLUMNS = {
    "country",
    "player_id",
    "slot",
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

    finite = numeric.dropna().to_numpy(
        dtype=float
    )

    if not np.isfinite(finite).all():
        raise ValueError(
            f"{column_name!r} contains non-finite values."
        )

    return numeric


def parse_roles(
    value: object,
) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()

    if isinstance(value, (list, tuple)):
        return tuple(
            str(role)
            for role in value
        )

    try:
        parsed = ast.literal_eval(
            str(value)
        )
    except (
        ValueError,
        SyntaxError,
    ):
        return ()

    if not isinstance(
        parsed,
        list,
    ):
        return ()

    return tuple(
        str(role)
        for role in parsed
    )


def global_zscore(
    values: pd.Series,
) -> pd.Series:
    numeric = finite_numeric(
        values,
        column_name="global z-score input",
    )

    mean = numeric.mean()
    standard_deviation = numeric.std(ddof=1)

    if (
        pd.isna(standard_deviation)
        or math.isclose(
            float(standard_deviation),
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        return pd.Series(
            0.0,
            index=values.index,
            dtype=float,
        )

    return (
        numeric
        - mean
    ) / standard_deviation


def winsorized_zscore(
    values: pd.Series,
) -> pd.Series:
    numeric = finite_numeric(
        values,
        column_name="winsorized z-score input",
    )

    valid = numeric.dropna()

    if valid.empty:
        return numeric

    lower = float(
        valid.quantile(
            WINSOR_LOWER_QUANTILE
        )
    )

    upper = float(
        valid.quantile(
            WINSOR_UPPER_QUANTILE
        )
    )

    clipped = numeric.clip(
        lower=lower,
        upper=upper,
    )

    return global_zscore(
        clipped
    )


def robust_zscore(
    values: pd.Series,
) -> pd.Series:
    numeric = finite_numeric(
        values,
        column_name="robust z-score input",
    )

    valid = numeric.dropna()

    if valid.empty:
        return numeric

    median = float(
        valid.median()
    )

    median_absolute_deviation = float(
        (
            valid
            - median
        )
        .abs()
        .median()
    )

    robust_scale = (
        ROBUST_SCALE_CONSTANT
        * median_absolute_deviation
    )

    if math.isclose(
        robust_scale,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return pd.Series(
            0.0,
            index=values.index,
            dtype=float,
        )

    return (
        numeric
        - median
    ) / robust_scale


def percentile_normal(
    values: pd.Series,
) -> pd.Series:
    """
    Convert empirical percentiles to an approximate standard-normal
    scale without allowing infinite tail values.
    """

    numeric = finite_numeric(
        values,
        column_name="percentile-normal input",
    )

    valid_mask = numeric.notna()

    result = pd.Series(
        np.nan,
        index=numeric.index,
        dtype=float,
    )

    valid = numeric.loc[
        valid_mask
    ]

    if valid.empty:
        return result

    percentiles = valid.rank(
        method="average",
        pct=True,
    )

    sample_size = len(valid)

    lower_bound = (
        0.5
        / max(
            sample_size,
            1,
        )
    )

    upper_bound = (
        1.0
        - lower_bound
    )

    clipped = percentiles.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    try:
        from scipy.stats import norm
    except ImportError as error:
        raise ImportError(
            "percentile_normal requires scipy."
        ) from error

    result.loc[
        valid_mask
    ] = norm.ppf(
        clipped.to_numpy(
            dtype=float
        )
    )

    return result


TRANSFORMATION_FUNCTIONS: dict[
    str,
    Callable[
        [pd.Series],
        pd.Series,
    ],
] = {
    "global_zscore":
        global_zscore,
    "winsorized_zscore":
        winsorized_zscore,
    "robust_zscore":
        robust_zscore,
    "percentile_normal":
        percentile_normal,
}


def load_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    features = pd.read_csv(
        FEATURES_PATH,
        dtype={
            "season_year": str,
        },
    )

    competitions = pd.read_csv(
        COMPETITION_PATH,
        dtype={
            "season_year": str,
        },
    )

    competition_features = pd.read_csv(
        COMPETITION_FEATURE_PATH,
        dtype={
            "season_year": str,
        },
    )

    registry = pd.read_csv(
        REGISTRY_PATH
    )

    feature_manifest = pd.read_csv(
        FEATURE_ATTRIBUTE_PATH
    )

    role_manifest = pd.read_csv(
        ROLE_ATTRIBUTE_PATH
    )

    expected_lineups = pd.read_csv(
        EXPECTED_LINEUPS_PATH
    )

    require_columns(
        features,
        REQUIRED_FEATURE_COLUMNS,
        frame_name="WC 2026 model features",
    )

    require_columns(
        competitions,
        REQUIRED_COMPETITION_COLUMNS,
        frame_name="Competition manifest",
    )

    require_columns(
        competition_features,
        REQUIRED_COMPETITION_FEATURE_COLUMNS,
        frame_name="Competition feature manifest",
    )

    require_columns(
        registry,
        REQUIRED_REGISTRY_COLUMNS,
        frame_name="Player registry",
    )

    require_columns(
        feature_manifest,
        REQUIRED_FEATURE_MANIFEST_COLUMNS,
        frame_name="Feature-attribute manifest",
    )

    require_columns(
        role_manifest,
        REQUIRED_ROLE_MANIFEST_COLUMNS,
        frame_name="Role-attribute manifest",
    )

    require_columns(
        expected_lineups,
        REQUIRED_LINEUP_COLUMNS,
        frame_name="Expected lineups",
    )

    return (
        features,
        competitions,
        competition_features,
        registry,
        feature_manifest,
        role_manifest,
        expected_lineups,
    )


def prepare_feature_rows(
    *,
    features: pd.DataFrame,
    competitions: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    output = features.merge(
        competitions[
            [
                "competition_id",
                "season_id",
                "recency_weight",
                "competition_importance",
            ]
        ],
        on=[
            "competition_id",
            "season_id",
        ],
        how="left",
        validate="many_to_one",
    )

    registry_subset = (
        registry[
            [
                "player_id",
                "canonical_player_id",
                "eligible_roles",
                "country",
            ]
        ]
        .drop_duplicates(
            subset=["player_id"]
        )
        .rename(
            columns={
                "eligible_roles":
                    "registry_eligible_roles",
                "country":
                    "registry_country",
            }
        )
    )

    output = output.merge(
        registry_subset,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    output[
        "canonical_player_id"
    ] = output[
        "canonical_player_id"
    ].fillna(
        output[
            "player_id"
        ]
    )

    if "eligible_roles" in output.columns:
        output["resolved_eligible_roles"] = (
            output["eligible_roles"]
            .where(
                output["eligible_roles"].notna(),
                output["registry_eligible_roles"],
            )
        )
    else:
        output["resolved_eligible_roles"] = (
            output["registry_eligible_roles"]
        )

    output[
        "eligible_roles_list"
    ] = output[
        "resolved_eligible_roles"
    ].apply(
        parse_roles
    )

    output[
        "minutesPlayed"
    ] = finite_numeric(
        output[
            "minutesPlayed"
        ],
        column_name="minutesPlayed",
    ).fillna(0.0)

    output[
        "recency_weight"
    ] = finite_numeric(
        output[
            "recency_weight"
        ],
        column_name="recency_weight",
    ).fillna(1.0)

    output[
        "competition_importance"
    ] = finite_numeric(
        output[
            "competition_importance"
        ],
        column_name="competition_importance",
    ).fillna(1.0)

    output["canonical_player_id"] = (
        output["canonical_player_id"]
        .fillna(output["player_id"])
    )

    if "eligible_roles" in output.columns:
        output["resolved_eligible_roles"] = (
            output["eligible_roles"]
            .combine_first(
                output["registry_eligible_roles"]
            )
        )
    else:
        output["resolved_eligible_roles"] = (
            output["registry_eligible_roles"]
        )

    if "country" in output.columns:
        output["resolved_country"] = (
            output["country"]
            .combine_first(
                output["registry_country"]
            )
        )
    else:
        output["resolved_country"] = (
            output["registry_country"]
        )

    output["resolved_eligible_roles"] = (
        output["resolved_eligible_roles"]
        .fillna("[]")
    )

    output["eligible_roles_list"] = (
        output["resolved_eligible_roles"]
        .apply(parse_roles)
    )

    required_resolved_columns = {
        "resolved_country",
        "resolved_eligible_roles",
        "eligible_roles_list",
    }

    missing_resolved_columns = (
        required_resolved_columns
        - set(output.columns)
    )

    if missing_resolved_columns:
        raise AssertionError(
            "Feature-row preparation failed to create resolved "
            f"registry fields: {sorted(missing_resolved_columns)}"
        )
    output[
        "row_weight"
    ] = (
        output[
            "minutesPlayed"
        ]
        * output[
            "recency_weight"
        ]
        * output[
            "competition_importance"
        ]
    )

    print(
        [
            column
            for column in output.columns
            if (
                "country" in column
                or "eligible_roles" in column
            )
        ]
    )

    return output


def add_transformed_features(
    *,
    feature_rows: pd.DataFrame,
    feature_manifest: pd.DataFrame,
) -> pd.DataFrame:
    output = feature_rows.copy()

    available_features = sorted(
        set(
            feature_manifest[
                "feature"
            ]
        )
        & set(
            output.columns
        )
    )

    if not available_features:
        raise ValueError(
            "No manifest features exist in the model-feature dataset."
        )

    for transformation_name in TRANSFORMATIONS:
        transformation_function = (
            TRANSFORMATION_FUNCTIONS[
                transformation_name
            ]
        )

        for feature in available_features:
            output[
                (
                    f"{transformation_name}"
                    f"__{feature}"
                )
            ] = transformation_function(
                output[
                    feature
                ]
            )

    return output


def feature_availability_lookup(
    competition_features: pd.DataFrame,
) -> dict[
    tuple[str, str, str],
    bool,
]:
    output = competition_features.copy()

    output[
        "competition"
    ] = output[
        "competition"
    ].fillna("").astype(str)

    output[
        "season_year"
    ] = output[
        "season_year"
    ].fillna("").astype(str)

    output[
        "available"
    ] = (
        output[
            "available"
        ]
        .fillna(True)
        .astype(bool)
    )

    return {
        (
            str(row.competition),
            str(row.season_year),
            str(row.feature),
        ):
            bool(row.available)
        for row in output.itertuples(
            index=False
        )
    }


def build_player_attributes(
    *,
    feature_rows: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    competition_features: pd.DataFrame,
) -> pd.DataFrame:
    manifest = feature_manifest.copy()

    manifest[
        "weight"
    ] = finite_numeric(
        manifest[
            "weight"
        ],
        column_name="feature manifest weight",
        allow_missing=False,
    )

    availability_lookup = (
        feature_availability_lookup(
            competition_features
        )
    )

    records: list[
        dict[str, object]
    ] = []

    grouped_players = feature_rows.groupby(
        "canonical_player_id",
        sort=True,
    )

    for canonical_player_id, group in grouped_players:
        base_record = {
            "canonical_player_id":
                canonical_player_id,
            "player_id":
                group[
                    "player_id"
                ].iloc[-1],
            "player":
                (
                    group[
                        "player"
                    ].dropna().iloc[-1]
                    if group[
                        "player"
                    ].notna().any()
                    else pd.NA
                ),
            "country":
                (
                    group[
                        "resolved_country"
                    ].dropna().iloc[-1]
                    if group[
                        "resolved_country"
                    ].notna().any()
                    else pd.NA
                ),
            "eligible_roles":
                (
                    group[
                        "resolved_eligible_roles"
                    ].dropna().iloc[-1]
                    if group[
                        "resolved_eligible_roles"
                    ].notna().any()
                    else "[]"
                ),
            "minutesPlayed":
                float(
                    group[
                        "minutesPlayed"
                    ].sum()
                ),
            "total_weighted_evidence":
                float(
                    group[
                        "row_weight"
                    ].sum()
                ),
        }

        evidence_confidence = min(
            float(
                base_record[
                    "total_weighted_evidence"
                ]
            ) / 1800.0,
            1.0,
        )

        for transformation_name in TRANSFORMATIONS:
            record = {
                **base_record,
                "transformation":
                    transformation_name,
                "evidence_confidence":
                    evidence_confidence,
            }

            for attribute, attribute_rows in (
                manifest.groupby(
                    "attribute",
                    sort=True,
                )
            ):
                score_parts: list[float] = []
                used_weights: list[float] = []

                for feature_row in (
                    attribute_rows.itertuples(
                        index=False
                    )
                ):
                    feature = str(
                        feature_row.feature
                    )

                    weight = float(
                        feature_row.weight
                    )

                    transformed_column = (
                        f"{transformation_name}"
                        f"__{feature}"
                    )

                    if (
                        transformed_column
                        not in group.columns
                    ):
                        continue

                    values = group[
                        transformed_column
                    ]

                    weights = group[
                        "row_weight"
                    ]

                    availability = pd.Series(
                        [
                            availability_lookup.get(
                                (
                                    str(
                                        competition
                                    ),
                                    str(
                                        season_year
                                    ),
                                    feature,
                                ),
                                True,
                            )
                            for competition, season_year
                            in zip(
                                group[
                                    "competition"
                                ].fillna(""),
                                group[
                                    "season_year"
                                ].fillna(""),
                                strict=True,
                            )
                        ],
                        index=group.index,
                        dtype=bool,
                    )

                    valid = (
                        values.notna()
                        & weights.notna()
                        & weights.gt(0.0)
                        & availability
                    )

                    if not valid.any():
                        continue

                    feature_score = float(
                        np.average(
                            values.loc[
                                valid
                            ].to_numpy(
                                dtype=float
                            ),
                            weights=weights.loc[
                                valid
                            ].to_numpy(
                                dtype=float
                            ),
                        )
                    )

                    score_parts.append(
                        feature_score
                        * weight
                    )

                    used_weights.append(
                        weight
                    )

                if score_parts:
                    record[
                        f"attribute_{attribute}"
                    ] = float(
                        sum(score_parts)
                        / sum(used_weights)
                    )

                else:
                    record[
                        f"attribute_{attribute}"
                    ] = np.nan

            records.append(record)

    return pd.DataFrame(
        records
    )


def build_player_roles(
    *,
    player_attributes: pd.DataFrame,
    role_manifest: pd.DataFrame,
) -> pd.DataFrame:
    manifest = role_manifest.copy()

    manifest[
        "weight"
    ] = finite_numeric(
        manifest[
            "weight"
        ],
        column_name="role manifest weight",
        allow_missing=False,
    )

    records: list[
        dict[str, object]
    ] = []

    for row in player_attributes.itertuples(
        index=False
    ):
        eligible_roles = set(
            parse_roles(
                row.eligible_roles
            )
        )

        for role in ROLES:
            if role not in eligible_roles:
                continue

            role_rows = manifest.loc[
                manifest[
                    "role"
                ].eq(role)
            ]

            score_parts: list[float] = []
            used_weights: list[float] = []

            for role_row in role_rows.itertuples(
                index=False
            ):
                attribute_column = (
                    f"attribute_"
                    f"{role_row.attribute}"
                )

                if not hasattr(
                    row,
                    attribute_column,
                ):
                    continue

                value = getattr(
                    row,
                    attribute_column,
                )

                if pd.isna(value):
                    continue

                weight = float(
                    role_row.weight
                )

                score_parts.append(
                    float(value)
                    * weight
                )

                used_weights.append(
                    weight
                )

            if not score_parts:
                continue

            raw_rating = float(
                sum(score_parts)
                / sum(used_weights)
            )

            adjusted_rating = (
                raw_rating
                * float(
                    row.evidence_confidence
                )
            )

            records.append(
                {
                    "canonical_player_id":
                        row.canonical_player_id,
                    "player_id":
                        row.player_id,
                    "player":
                        row.player,
                    "country":
                        row.country,
                    "transformation":
                        row.transformation,
                    "role":
                        role,
                    "raw_rating":
                        raw_rating,
                    "adjusted_rating":
                        adjusted_rating,
                    "evidence_confidence":
                        float(
                            row.evidence_confidence
                        ),
                    "minutesPlayed":
                        float(
                            row.minutesPlayed
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


def summarize_values(
    values: pd.Series,
) -> dict[str, object]:
    numeric = finite_numeric(
        values,
        column_name="summary values",
        allow_missing=False,
    )

    return {
        "count":
            len(numeric),
        "minimum":
            float(
                numeric.min()
            ),
        "p01":
            float(
                numeric.quantile(0.01)
            ),
        "p05":
            float(
                numeric.quantile(0.05)
            ),
        "p25":
            float(
                numeric.quantile(0.25)
            ),
        "median":
            float(
                numeric.median()
            ),
        "mean":
            float(
                numeric.mean()
            ),
        "p75":
            float(
                numeric.quantile(0.75)
            ),
        "p95":
            float(
                numeric.quantile(0.95)
            ),
        "p99":
            float(
                numeric.quantile(0.99)
            ),
        "maximum":
            float(
                numeric.max()
            ),
        "standard_deviation":
            float(
                numeric.std(
                    ddof=0
                )
            ),
        "interquartile_range":
            float(
                numeric.quantile(0.75)
                - numeric.quantile(0.25)
            ),
        "negative_count":
            int(
                numeric.lt(0.0).sum()
            ),
    }


def build_feature_distribution_summary(
    *,
    feature_rows: pd.DataFrame,
    feature_manifest: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    available_features = sorted(
        set(
            feature_manifest[
                "feature"
            ]
        )
        & set(
            feature_rows.columns
        )
    )

    for transformation_name in TRANSFORMATIONS:
        for feature in available_features:
            column = (
                f"{transformation_name}"
                f"__{feature}"
            )

            values = feature_rows[
                column
            ].dropna()

            if values.empty:
                continue

            records.append(
                {
                    "transformation":
                        transformation_name,
                    "feature":
                        feature,
                    **summarize_values(
                        values
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_attribute_distribution_summary(
    player_attributes: pd.DataFrame,
) -> pd.DataFrame:
    attribute_columns = sorted(
        column
        for column
        in player_attributes.columns
        if column.startswith(
            "attribute_"
        )
    )

    records: list[
        dict[str, object]
    ] = []

    for transformation_name, rows in (
        player_attributes.groupby(
            "transformation",
            sort=True,
        )
    ):
        for column in attribute_columns:
            values = rows[
                column
            ].dropna()

            if values.empty:
                continue

            records.append(
                {
                    "transformation":
                        transformation_name,
                    "attribute":
                        column.replace(
                            "attribute_",
                            "",
                            1,
                        ),
                    **summarize_values(
                        values
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_role_distribution_summary(
    player_roles: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for (
        transformation_name,
        role,
    ), rows in player_roles.groupby(
        [
            "transformation",
            "role",
        ],
        sort=True,
    ):
        for scale_name, column in (
            (
                "raw",
                "raw_rating",
            ),
            (
                "evidence_adjusted",
                "adjusted_rating",
            ),
        ):
            records.append(
                {
                    "transformation":
                        transformation_name,
                    "role":
                        role,
                    "scale":
                        scale_name,
                    "mean_evidence_confidence":
                        float(
                            rows[
                                "evidence_confidence"
                            ].mean()
                        ),
                    **summarize_values(
                        rows[
                            column
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_role_scale_compatibility(
    role_distributions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    adjusted = role_distributions.loc[
        role_distributions[
            "scale"
        ].eq(
            "evidence_adjusted"
        )
    ]

    for transformation_name, rows in (
        adjusted.groupby(
            "transformation",
            sort=True,
        )
    ):
        by_role = rows.set_index(
            "role"
        )

        goalkeeper_mean = float(
            by_role.at[
                "GK",
                "mean",
            ]
        )

        goalkeeper_median = float(
            by_role.at[
                "GK",
                "median",
            ]
        )

        goalkeeper_std = float(
            by_role.at[
                "GK",
                "standard_deviation",
            ]
        )

        outfield = by_role.drop(
            index="GK"
        )

        outfield_mean = float(
            outfield[
                "mean"
            ].mean()
        )

        outfield_median = float(
            outfield[
                "median"
            ].mean()
        )

        outfield_std = float(
            outfield[
                "standard_deviation"
            ].mean()
        )

        records.append(
            {
                "transformation":
                    transformation_name,
                "goalkeeper_mean":
                    goalkeeper_mean,
                "mean_outfield_role_mean":
                    outfield_mean,
                "goalkeeper_to_outfield_mean_ratio":
                    (
                        goalkeeper_mean
                        / outfield_mean
                        if not math.isclose(
                            outfield_mean,
                            0.0,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        else None
                    ),
                "goalkeeper_median":
                    goalkeeper_median,
                "mean_outfield_role_median":
                    outfield_median,
                "goalkeeper_to_outfield_median_ratio":
                    (
                        goalkeeper_median
                        / outfield_median
                        if not math.isclose(
                            outfield_median,
                            0.0,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        else None
                    ),
                "goalkeeper_standard_deviation":
                    goalkeeper_std,
                "mean_outfield_role_standard_deviation":
                    outfield_std,
                "goalkeeper_to_outfield_std_ratio":
                    (
                        goalkeeper_std
                        / outfield_std
                        if not math.isclose(
                            outfield_std,
                            0.0,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        else None
                    ),
                "minimum_role_mean":
                    float(
                        by_role[
                            "mean"
                        ].min()
                    ),
                "maximum_role_mean":
                    float(
                        by_role[
                            "mean"
                        ].max()
                    ),
                "role_mean_range":
                    float(
                        by_role[
                            "mean"
                        ].max()
                        - by_role[
                            "mean"
                        ].min()
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def weighted_role_projection(
    role_values: dict[str, float],
    weights: dict[str, float],
) -> float:
    total_weight = float(
        sum(
            weights.values()
        )
    )

    if total_weight <= 0.0:
        raise ValueError(
            "Projection weights must sum to a positive value."
        )

    return float(
        sum(
            role_values.get(
                role,
                0.0,
            )
            * weight
            for role, weight
            in weights.items()
        )
        / total_weight
    )


def build_expected_xi_dimensions(
    *,
    expected_lineups: pd.DataFrame,
    player_roles: pd.DataFrame,
) -> pd.DataFrame:
    lineups = expected_lineups.copy()

    lineups[
        "player_id"
    ] = pd.to_numeric(
        lineups[
            "player_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    selected = lineups.loc[
        lineups[
            "player_id"
        ].notna()
    ].copy()

    complete_teams = set(
        selected.groupby(
            "country"
        )[
            "player_id"
        ]
        .nunique()
        .loc[
            lambda values:
                values.eq(
                    EXPECTED_XI_REQUIRED_PLAYERS
                )
        ]
        .index
    )

    selected = selected.loc[
        selected[
            "country"
        ].isin(
            complete_teams
        )
    ].copy()

    role_lookup = (
        player_roles.pivot_table(
            index=[
                "transformation",
                "player_id",
            ],
            columns="role",
            values="adjusted_rating",
            aggfunc="first",
        )
        .reset_index()
    )

    merged = selected.merge(
        role_lookup,
        on="player_id",
        how="left",
        validate="many_to_many",
    )

    records: list[
        dict[str, object]
    ] = []

    for (
        transformation_name,
        country,
    ), rows in merged.groupby(
        [
            "transformation",
            "country",
        ],
        sort=True,
    ):
        if rows[
            "player_id"
        ].nunique() != 11:
            continue

        attack_values: list[float] = []
        midfield_values: list[float] = []
        defense_values: list[float] = []
        goalkeeper_values: list[float] = []

        for row in rows.itertuples(
            index=False
        ):
            role_values = {
                role: (
                    0.0
                    if pd.isna(
                        getattr(
                            row,
                            role,
                            np.nan,
                        )
                    )
                    else float(
                        getattr(
                            row,
                            role,
                        )
                    )
                )
                for role in ROLES
            }

            attack_values.append(
                weighted_role_projection(
                    role_values,
                    PROJECTION_WEIGHTS[
                        "attack"
                    ],
                )
            )

            midfield_values.append(
                weighted_role_projection(
                    role_values,
                    PROJECTION_WEIGHTS[
                        "midfield"
                    ],
                )
            )

            defense_values.append(
                weighted_role_projection(
                    role_values,
                    PROJECTION_WEIGHTS[
                        "defense"
                    ],
                )
            )

            goalkeeper_values.append(
                weighted_role_projection(
                    role_values,
                    PROJECTION_WEIGHTS[
                        "goalkeeper"
                    ],
                )
            )

        records.append(
            {
                "transformation":
                    transformation_name,
                "country":
                    country,
                "attack":
                    float(
                        np.mean(
                            sorted(
                                attack_values,
                                reverse=True,
                            )[:5]
                        )
                    ),
                "midfield":
                    float(
                        np.mean(
                            sorted(
                                midfield_values,
                                reverse=True,
                            )[:5]
                        )
                    ),
                "defense":
                    float(
                        np.mean(
                            sorted(
                                defense_values,
                                reverse=True,
                            )[:5]
                        )
                    ),
                "goalkeeper":
                    float(
                        max(
                            goalkeeper_values
                        )
                    ),
                "player_count":
                    len(rows),
            }
        )

    return pd.DataFrame(
        records
    )


def build_team_scale_summary(
    dimensions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for transformation_name, rows in (
        dimensions.groupby(
            "transformation",
            sort=True,
        )
    ):
        means = {
            dimension: float(
                rows[
                    dimension
                ].mean()
            )
            for dimension in (
                "attack",
                "midfield",
                "defense",
                "goalkeeper",
            )
        }

        outfield_mean = float(
            np.mean(
                [
                    means[
                        "attack"
                    ],
                    means[
                        "midfield"
                    ],
                    means[
                        "defense"
                    ],
                ]
            )
        )

        records.append(
            {
                "transformation":
                    transformation_name,
                "team_count":
                    rows[
                        "country"
                    ].nunique(),
                "attack_mean":
                    means[
                        "attack"
                    ],
                "midfield_mean":
                    means[
                        "midfield"
                    ],
                "defense_mean":
                    means[
                        "defense"
                    ],
                "goalkeeper_mean":
                    means[
                        "goalkeeper"
                    ],
                "mean_outfield_dimension":
                    outfield_mean,
                "goalkeeper_to_outfield_mean_ratio":
                    (
                        means[
                            "goalkeeper"
                        ]
                        / outfield_mean
                        if not math.isclose(
                            outfield_mean,
                            0.0,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        else None
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def validate_outputs(
    *,
    player_attributes: pd.DataFrame,
    player_roles: pd.DataFrame,
    feature_distributions: pd.DataFrame,
    attribute_distributions: pd.DataFrame,
    role_distributions: pd.DataFrame,
    role_compatibility: pd.DataFrame,
    expected_xi_dimensions: pd.DataFrame,
    team_scale_summary: pd.DataFrame,
) -> None:
    if player_attributes.empty:
        raise AssertionError(
            "Player attribute output is empty."
        )

    if player_roles.empty:
        raise AssertionError(
            "Player role output is empty."
        )

    if feature_distributions.empty:
        raise AssertionError(
            "Feature distribution output is empty."
        )

    if attribute_distributions.empty:
        raise AssertionError(
            "Attribute distribution output is empty."
        )

    if role_distributions.empty:
        raise AssertionError(
            "Role distribution output is empty."
        )

    if len(
        role_compatibility
    ) != len(
        TRANSFORMATIONS
    ):
        raise AssertionError(
            "Role compatibility output does not cover "
            "every transformation."
        )

    if expected_xi_dimensions.empty:
        raise AssertionError(
            "Expected-XI dimension output is empty."
        )

    if len(
        team_scale_summary
    ) != len(
        TRANSFORMATIONS
    ):
        raise AssertionError(
            "Team scale summary does not cover every transformation."
        )

    duplicate_attributes = (
        player_attributes[
            [
                "canonical_player_id",
                "transformation",
            ]
        ]
        .duplicated()
        .any()
    )

    if duplicate_attributes:
        raise AssertionError(
            "Player attributes contain duplicate "
            "player-transformation rows."
        )

    duplicate_roles = (
        player_roles[
            [
                "canonical_player_id",
                "transformation",
                "role",
            ]
        ]
        .duplicated()
        .any()
    )

    if duplicate_roles:
        raise AssertionError(
            "Player roles contain duplicate rows."
        )

    for frame_name, frame in {
        "feature distributions":
            feature_distributions,
        "attribute distributions":
            attribute_distributions,
        "role distributions":
            role_distributions,
        "role compatibility":
            role_compatibility,
        "expected-XI dimensions":
            expected_xi_dimensions,
        "team scale summary":
            team_scale_summary,
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
                f"{frame_name} contains non-finite values."
            )


def build_metadata(
    *,
    player_attributes: pd.DataFrame,
    player_roles: pd.DataFrame,
    role_compatibility: pd.DataFrame,
    expected_xi_dimensions: pd.DataFrame,
    team_scale_summary: pd.DataFrame,
) -> dict[str, Any]:
    global_z_row = role_compatibility.loc[
        role_compatibility[
            "transformation"
        ].eq(
            "global_zscore"
        )
    ].iloc[0]

    best_role_scale_row = (
        role_compatibility
        .assign(
            distance_from_one=lambda frame:
                (
                    frame[
                        "goalkeeper_to_outfield_mean_ratio"
                    ]
                    - 1.0
                ).abs()
        )
        .sort_values(
            [
                "distance_from_one",
                "role_mean_range",
            ]
        )
        .iloc[0]
    )

    best_team_scale_row = (
        team_scale_summary
        .assign(
            distance_from_one=lambda frame:
                (
                    frame[
                        "goalkeeper_to_outfield_mean_ratio"
                    ]
                    - 1.0
                ).abs()
        )
        .sort_values(
            "distance_from_one"
        )
        .iloc[0]
    )

    return {
        "study_id": "092B",
        "study_name": (
            "Feature Transformation Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "transformation_count":
            len(TRANSFORMATIONS),
        "transformations":
            list(
                TRANSFORMATIONS
            ),
        "player_attribute_row_count":
            len(player_attributes),
        "player_role_row_count":
            len(player_roles),
        "expected_xi_dimension_row_count":
            len(expected_xi_dimensions),
        "global_zscore_role_gk_to_outfield_ratio":
            float(
                global_z_row[
                    "goalkeeper_to_outfield_mean_ratio"
                ]
            ),
        "closest_role_scale_transformation":
            str(
                best_role_scale_row[
                    "transformation"
                ]
            ),
        "closest_role_scale_ratio":
            float(
                best_role_scale_row[
                    "goalkeeper_to_outfield_mean_ratio"
                ]
            ),
        "closest_team_scale_transformation":
            str(
                best_team_scale_row[
                    "transformation"
                ]
            ),
        "closest_team_scale_ratio":
            float(
                best_team_scale_row[
                    "goalkeeper_to_outfield_mean_ratio"
                ]
            ),
        "production_transformation_selected":
            False,
        "player_attribute_file_changed":
            False,
        "player_rating_file_changed":
            False,
        "team_repository_generated":
            False,
        "goal_model_fitted":
            False,
        "production_runtime_changed":
            False,
        "interpretation_boundary": (
            "This study compares feature transformations by their "
            "downstream distributional consequences. Proximity of "
            "cross-role scales does not establish predictive superiority."
        ),
        "outputs": [
            FEATURE_DISTRIBUTION_PATH.name,
            ATTRIBUTE_DISTRIBUTION_PATH.name,
            ROLE_DISTRIBUTION_PATH.name,
            ROLE_COMPARISON_PATH.name,
            PLAYER_ATTRIBUTE_PATH.name,
            PLAYER_ROLE_PATH.name,
            EXPECTED_XI_DIMENSION_PATH.name,
            TEAM_SCALE_SUMMARY_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    role_compatibility: pd.DataFrame,
    team_scale_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    report = f"""# Study 092B — Feature Transformation Audit

## Purpose

Compare alternative feature transformations and trace their effects
through:

```text
feature
→ player attribute
→ player role rating
→ expected-XI team dimension

Transformations

    {json.dumps(list(TRANSFORMATIONS), indent=2)}

Methodological controls

All candidates preserve:

the existing player identity resolution;
minutes × recency × competition-importance evidence weights;
competition-feature availability filtering;
the feature-to-attribute manifest;
the role-to-attribute manifest;
the existing evidence-confidence formula;
the existing role-projection weights;
top-five outfield aggregation;
best-goalkeeper aggregation.

Only the feature transformation changes.

Player-role scale compatibility

    {json.dumps(
    role_compatibility.to_dict(
    orient="records"
    ),
    indent=2,
    default=str,
    )}

Expected-XI team-dimension scale compatibility

    {json.dumps(
    team_scale_summary.to_dict(
    orient="records"
    ),
    indent=2,
    default=str,
    )}

Descriptive findings
Global z-score role GK/outfield ratio:
    {metadata["global_zscore_role_gk_to_outfield_ratio"]}
Transformation closest to equal role means:
    {metadata["closest_role_scale_transformation"]}
Its role-scale ratio:
    {metadata["closest_role_scale_ratio"]}
Transformation closest to equal expected-XI dimension means:
    {metadata["closest_team_scale_transformation"]}
Its team-scale ratio:
    {metadata["closest_team_scale_ratio"]}
Interpretation boundary

A ratio closer to one means the transformation produces more comparable
average numerical scales.

It does not prove that:

equal numerical values imply equal football quality;
the transformation improves prediction;
extreme observations should be removed;
the resulting team representation is production-ready.

Scale compatibility is necessary for direct weighted addition, but it is
not sufficient for predictive validity.

Methodological boundary

This study:

does not overwrite player_attribute_scores.csv;
does not overwrite player_ratings.csv;
does not change feature manifests;
does not change role manifests;
creates no production team repository;
fits no goal model;
chooses no winning transformation;
changes no simulation behavior.
Result

OVERALL RESULT: {metadata["status"]}

Alternative feature transformations were compared without changing the
production player-intelligence pipeline.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
    "STUDY 092B — FEATURE TRANSFORMATION AUDIT"
    )
    print("=" * 88)

    (
        features,
        competitions,
        competition_features,
        registry,
        feature_manifest,
        role_manifest,
        expected_lineups,
    ) = load_inputs()

    feature_rows = prepare_feature_rows(
        features=features,
        competitions=competitions,
        registry=registry,
    )

    transformed_feature_rows = (
        add_transformed_features(
            feature_rows=feature_rows,
            feature_manifest=(
                feature_manifest
            ),
        )
    )

    player_attributes = (
        build_player_attributes(
            feature_rows=(
                transformed_feature_rows
            ),
            feature_manifest=(
                feature_manifest
            ),
            competition_features=(
                competition_features
            ),
        )
    )

    player_roles = build_player_roles(
        player_attributes=(
            player_attributes
        ),
        role_manifest=role_manifest,
    )

    feature_distributions = (
        build_feature_distribution_summary(
            feature_rows=(
                transformed_feature_rows
            ),
            feature_manifest=(
                feature_manifest
            ),
        )
    )

    attribute_distributions = (
        build_attribute_distribution_summary(
            player_attributes
        )
    )

    role_distributions = (
        build_role_distribution_summary(
            player_roles
        )
    )

    role_compatibility = (
        build_role_scale_compatibility(
            role_distributions
        )
    )

    expected_xi_dimensions = (
        build_expected_xi_dimensions(
            expected_lineups=expected_lineups,
            player_roles=player_roles,
        )
    )

    team_scale_summary = (
        build_team_scale_summary(
            expected_xi_dimensions
        )
    )

    validate_outputs(
        player_attributes=player_attributes,
        player_roles=player_roles,
        feature_distributions=(
            feature_distributions
        ),
        attribute_distributions=(
            attribute_distributions
        ),
        role_distributions=(
            role_distributions
        ),
        role_compatibility=(
            role_compatibility
        ),
        expected_xi_dimensions=(
            expected_xi_dimensions
        ),
        team_scale_summary=(
            team_scale_summary
        ),
    )

    metadata = build_metadata(
        player_attributes=player_attributes,
        player_roles=player_roles,
        role_compatibility=(
            role_compatibility
        ),
        expected_xi_dimensions=(
            expected_xi_dimensions
        ),
        team_scale_summary=(
            team_scale_summary
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_distributions.to_csv(
        FEATURE_DISTRIBUTION_PATH,
        index=False,
    )

    attribute_distributions.to_csv(
        ATTRIBUTE_DISTRIBUTION_PATH,
        index=False,
    )

    role_distributions.to_csv(
        ROLE_DISTRIBUTION_PATH,
        index=False,
    )

    role_compatibility.to_csv(
        ROLE_COMPARISON_PATH,
        index=False,
    )

    player_attributes.to_csv(
        PLAYER_ATTRIBUTE_PATH,
        index=False,
    )

    player_roles.to_csv(
        PLAYER_ROLE_PATH,
        index=False,
    )

    expected_xi_dimensions.to_csv(
        EXPECTED_XI_DIMENSION_PATH,
        index=False,
    )

    team_scale_summary.to_csv(
        TEAM_SCALE_SUMMARY_PATH,
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
        role_compatibility=(
            role_compatibility
        ),
        team_scale_summary=(
            team_scale_summary
        ),
        metadata=metadata,
    )

    print()
    print("Player-role scale compatibility")
    print("-" * 88)
    print(
        role_compatibility[
            [
                "transformation",
                "goalkeeper_to_outfield_mean_ratio",
                "goalkeeper_to_outfield_median_ratio",
                "goalkeeper_to_outfield_std_ratio",
                "role_mean_range",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Expected-XI dimension scale compatibility")
    print("-" * 88)
    print(
        team_scale_summary[
            [
                "transformation",
                "team_count",
                "attack_mean",
                "midfield_mean",
                "defense_mean",
                "goalkeeper_mean",
                "goalkeeper_to_outfield_mean_ratio",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Feature transforms: PASS")
    print("  Attribute reconstruction: PASS")
    print("  Role reconstruction: PASS")
    print("  Expected-XI reconstruction: PASS")
    print("  Production files overwritten: NO")
    print("  Production transformation selected: NO")
    print("  Goal-model fitting: NONE")
    print("  Production runtime change: NONE")

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