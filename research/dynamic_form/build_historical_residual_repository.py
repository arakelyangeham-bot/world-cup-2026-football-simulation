#build_historical_residual_repository

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_064_prequential_goal_expectations"
    / "historical_goal_expectations.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_065_historical_residual_repository"
)

RESIDUAL_PATH = (
    OUTPUT_DIRECTORY
    / "historical_residual_repository.csv"
)

MATCH_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "residual_identity_audit.csv"
)

TEAM_RESIDUAL_PATH = (
    OUTPUT_DIRECTORY
    / "team_residual_history.csv"
)

DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "residual_distribution_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


REQUIRED_COLUMNS = {
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
}

RESIDUAL_COLUMNS = (
    "home_attack_residual",
    "home_defense_residual",
    "away_attack_residual",
    "away_defense_residual",
)


def parse_boolean(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    parsed = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    if parsed.isna().any():
        invalid = sorted(
            normalized[
                parsed.isna()
            ].unique().tolist()
        )

        raise ValueError(
            f"{column_name} contains invalid Boolean "
            f"values: {invalid}"
        )

    return parsed.astype(bool)


def load_expectations() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Study 064 expectation dataset does not "
            f"exist: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 064 expectation dataset is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Expectation dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    for column in (
        "date",
        "training_start_date",
        "training_end_date",
        "prediction_date",
    ):
        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
            utc=True,
        )

    if dataframe["date"].isna().any():
        raise ValueError(
            "Expectation dataset contains invalid match "
            "dates."
        )

    if dataframe["prediction_date"].isna().any():
        raise ValueError(
            "Expectation dataset contains invalid "
            "prediction dates."
        )

    dataframe["prediction_available"] = (
        parse_boolean(
            dataframe["prediction_available"],
            "prediction_available",
        )
    )

    dataframe[
        "prediction_temporal_validity_pass"
    ] = parse_boolean(
        dataframe[
            "prediction_temporal_validity_pass"
        ],
        "prediction_temporal_validity_pass",
    )

    numeric_columns = (
        "actual_home_goals",
        "actual_away_goals",
        "expected_home_goals",
        "expected_away_goals",
        "training_match_count",
        "model_alpha",
    )

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if dataframe["event_id"].isna().any():
        raise ValueError(
            "Expectation dataset contains missing "
            "event IDs."
        )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Expectation dataset contains duplicate "
            "event IDs."
        )

    if dataframe[
        [
            "actual_home_goals",
            "actual_away_goals",
            "training_match_count",
            "model_alpha",
        ]
    ].isna().any().any():
        raise ValueError(
            "Expectation dataset contains missing "
            "required numeric values."
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


def validate_expectation_contract(
    expectations: pd.DataFrame,
) -> None:
    available = expectations[
        "prediction_available"
    ]

    unavailable = ~available

    if not available.any():
        raise AssertionError(
            "No available expectations exist."
        )

    if expectations.loc[
        available,
        [
            "expected_home_goals",
            "expected_away_goals",
        ],
    ].isna().any().any():
        raise AssertionError(
            "Available rows contain missing expected "
            "goals."
        )

    if not np.isfinite(
        expectations.loc[
            available,
            [
                "expected_home_goals",
                "expected_away_goals",
            ],
        ].to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            "Available expected goals contain "
            "non-finite values."
        )

    if (
        expectations.loc[
            available,
            [
                "expected_home_goals",
                "expected_away_goals",
            ],
        ]
        <= 0.0
    ).any().any():
        raise AssertionError(
            "Available expected goals must be positive."
        )

    if expectations.loc[
        unavailable,
        [
            "expected_home_goals",
            "expected_away_goals",
        ],
    ].notna().any().any():
        raise AssertionError(
            "Unavailable warm-up rows unexpectedly "
            "contain expected goals."
        )

    if not expectations.loc[
        available,
        "prediction_temporal_validity_pass",
    ].all():
        raise AssertionError(
            "An available expectation failed temporal "
            "validation."
        )

    available_training_end = pd.to_datetime(
        expectations.loc[
            available,
            "training_end_date",
        ],
        errors="raise",
        utc=True,
    )

    available_prediction_dates = pd.to_datetime(
        expectations.loc[
            available,
            "prediction_date",
        ],
        errors="raise",
        utc=True,
    )

    if not (
        available_training_end.dt.date
        < available_prediction_dates.dt.date
    ).all():
        raise AssertionError(
            "An available expectation used same-day or "
            "future training information."
        )


def build_residual_repository(
    expectations: pd.DataFrame,
) -> pd.DataFrame:
    output = expectations.copy()

    available = output[
        "prediction_available"
    ]

    output["home_attack_residual"] = np.nan
    output["home_defense_residual"] = np.nan
    output["away_attack_residual"] = np.nan
    output["away_defense_residual"] = np.nan

    output.loc[
        available,
        "home_attack_residual",
    ] = (
        output.loc[
            available,
            "actual_home_goals",
        ]
        - output.loc[
            available,
            "expected_home_goals",
        ]
    )

    output.loc[
        available,
        "home_defense_residual",
    ] = (
        output.loc[
            available,
            "expected_away_goals",
        ]
        - output.loc[
            available,
            "actual_away_goals",
        ]
    )

    output.loc[
        available,
        "away_attack_residual",
    ] = (
        output.loc[
            available,
            "actual_away_goals",
        ]
        - output.loc[
            available,
            "expected_away_goals",
        ]
    )

    output.loc[
        available,
        "away_defense_residual",
    ] = (
        output.loc[
            available,
            "expected_home_goals",
        ]
        - output.loc[
            available,
            "actual_home_goals",
        ]
    )

    output["residual_available"] = available

    output["residual_source"] = (
        "prequential_goal_expectations"
    )

    output["residual_definition_version"] = (
        "1.0"
    )

    return output


def validate_residual_identity(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    available = residuals[
        "residual_available"
    ]

    unavailable = ~available

    if residuals.loc[
        available,
        list(RESIDUAL_COLUMNS),
    ].isna().any().any():
        raise AssertionError(
            "Available residual rows contain missing "
            "values."
        )

    if residuals.loc[
        unavailable,
        list(RESIDUAL_COLUMNS),
    ].notna().any().any():
        raise AssertionError(
            "Unavailable rows unexpectedly contain "
            "residual values."
        )

    expected_values = {
        "home_attack_residual": (
            residuals["actual_home_goals"]
            - residuals["expected_home_goals"]
        ),
        "home_defense_residual": (
            residuals["expected_away_goals"]
            - residuals["actual_away_goals"]
        ),
        "away_attack_residual": (
            residuals["actual_away_goals"]
            - residuals["expected_away_goals"]
        ),
        "away_defense_residual": (
            residuals["expected_home_goals"]
            - residuals["actual_home_goals"]
        ),
    }

    audit = residuals[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "prediction_available",
            "residual_available",
        ]
    ].copy()

    identity_columns: list[str] = []

    for residual_column, expected in (
        expected_values.items()
    ):
        identity_column = (
            f"{residual_column}_identity_pass"
        )

        audit[identity_column] = np.where(
            available,
            np.isclose(
                residuals[
                    residual_column
                ].to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
                equal_nan=False,
            ),
            residuals[
                residual_column
            ].isna(),
        )

        identity_columns.append(
            identity_column
        )

    audit[
        "availability_contract_pass"
    ] = (
        residuals["residual_available"]
        .eq(
            residuals["prediction_available"]
        )
    )

    audit["overall_validation_pass"] = (
        audit[identity_columns]
        .all(axis=1)
        & audit[
            "availability_contract_pass"
        ]
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
            "One or more residual identities failed."
        )

    return audit


def build_team_residual_history(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    available = residuals[
        residuals[
            "residual_available"
        ]
    ].copy()

    home = pd.DataFrame(
        {
            "event_id":
                available["event_id"],
            "date":
                available["date"],
            "team_id":
                available["home_team_id"],
            "team_name":
                available["home_team"],
            "opponent_id":
                available["away_team_id"],
            "opponent_name":
                available["away_team"],
            "venue_role":
                "home",
            "goals_scored":
                available["actual_home_goals"],
            "goals_conceded":
                available["actual_away_goals"],
            "expected_goals_scored":
                available["expected_home_goals"],
            "expected_goals_conceded":
                available["expected_away_goals"],
            "attack_residual":
                available[
                    "home_attack_residual"
                ],
            "defense_residual":
                available[
                    "home_defense_residual"
                ],
            "baseline_name":
                available["baseline_name"],
            "baseline_version":
                available["baseline_version"],
            "feature_specification":
                available[
                    "feature_specification"
                ],
            "training_match_count":
                available[
                    "training_match_count"
                ],
            "prediction_temporal_validity_pass":
                available[
                    "prediction_temporal_validity_pass"
                ],
        }
    )

    away = pd.DataFrame(
        {
            "event_id":
                available["event_id"],
            "date":
                available["date"],
            "team_id":
                available["away_team_id"],
            "team_name":
                available["away_team"],
            "opponent_id":
                available["home_team_id"],
            "opponent_name":
                available["home_team"],
            "venue_role":
                "away",
            "goals_scored":
                available["actual_away_goals"],
            "goals_conceded":
                available["actual_home_goals"],
            "expected_goals_scored":
                available["expected_away_goals"],
            "expected_goals_conceded":
                available["expected_home_goals"],
            "attack_residual":
                available[
                    "away_attack_residual"
                ],
            "defense_residual":
                available[
                    "away_defense_residual"
                ],
            "baseline_name":
                available["baseline_name"],
            "baseline_version":
                available["baseline_version"],
            "feature_specification":
                available[
                    "feature_specification"
                ],
            "training_match_count":
                available[
                    "training_match_count"
                ],
            "prediction_temporal_validity_pass":
                available[
                    "prediction_temporal_validity_pass"
                ],
        }
    )

    output = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    output = (
        output
        .sort_values(
            [
                "date",
                "event_id",
                "venue_role",
            ]
        )
        .reset_index(drop=True)
    )

    expected_rows = (
        2 * int(
            residuals[
                "residual_available"
            ].sum()
        )
    )

    if len(output) != expected_rows:
        raise AssertionError(
            "Team residual history has an unexpected "
            f"row count: {len(output)} vs "
            f"{expected_rows}."
        )

    if output[
        [
            "event_id",
            "team_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Team residual history contains duplicate "
            "event/team records."
        )

    if output[
        [
            "attack_residual",
            "defense_residual",
        ]
    ].isna().any().any():
        raise AssertionError(
            "Team residual history contains missing "
            "residual values."
        )

    if not output[
        "prediction_temporal_validity_pass"
    ].all():
        raise AssertionError(
            "Team residual history contains a "
            "temporally invalid record."
        )

    return output


def summarize_series(
    label: str,
    series: pd.Series,
) -> dict[str, object]:
    values = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} contains non-finite values."
        )

    return {
        "residual_type": label,
        "observation_count": len(values),
        "mean": float(
            np.mean(values)
        ),
        "standard_deviation": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "minimum": float(
            np.min(values)
        ),
        "first_quartile": float(
            np.quantile(
                values,
                0.25,
            )
        ),
        "median": float(
            np.median(values)
        ),
        "third_quartile": float(
            np.quantile(
                values,
                0.75,
            )
        ),
        "maximum": float(
            np.max(values)
        ),
        "mean_absolute_residual": float(
            np.mean(
                np.abs(values)
            )
        ),
        "root_mean_squared_residual": float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        ),
    }


def build_distribution_summary(
    residuals: pd.DataFrame,
    team_history: pd.DataFrame,
) -> pd.DataFrame:
    available = residuals[
        residuals[
            "residual_available"
        ]
    ]

    records = [
        summarize_series(
            "home_attack_residual",
            available[
                "home_attack_residual"
            ],
        ),
        summarize_series(
            "home_defense_residual",
            available[
                "home_defense_residual"
            ],
        ),
        summarize_series(
            "away_attack_residual",
            available[
                "away_attack_residual"
            ],
        ),
        summarize_series(
            "away_defense_residual",
            available[
                "away_defense_residual"
            ],
        ),
        summarize_series(
            "all_team_attack_residuals",
            team_history[
                "attack_residual"
            ],
        ),
        summarize_series(
            "all_team_defense_residuals",
            team_history[
                "defense_residual"
            ],
        ),
    ]

    return pd.DataFrame(records)


def validate_cross_perspective_identities(
    residuals: pd.DataFrame,
) -> None:
    available = residuals[
        residuals[
            "residual_available"
        ]
    ]

    if not np.allclose(
        available[
            "home_defense_residual"
        ].to_numpy(dtype=float),
        -available[
            "away_attack_residual"
        ].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Home defensive residual does not equal "
            "negative away attacking residual."
        )

    if not np.allclose(
        available[
            "away_defense_residual"
        ].to_numpy(dtype=float),
        -available[
            "home_attack_residual"
        ].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Away defensive residual does not equal "
            "negative home attacking residual."
        )


def build_metadata(
    expectations: pd.DataFrame,
    residuals: pd.DataFrame,
    team_history: pd.DataFrame,
    distribution_summary: pd.DataFrame,
) -> dict[str, object]:
    available = residuals[
        residuals[
            "residual_available"
        ]
    ]

    attack_summary = (
        distribution_summary[
            distribution_summary[
                "residual_type"
            ].eq(
                "all_team_attack_residuals"
            )
        ].iloc[0]
    )

    defense_summary = (
        distribution_summary[
            distribution_summary[
                "residual_type"
            ].eq(
                "all_team_defense_residuals"
            )
        ].iloc[0]
    )

    return {
        "study_id": "065",
        "study_name": (
            "Historical Dynamic-Form Residual "
            "Repository"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_path": str(INPUT_PATH),
        "residual_output_path": str(
            RESIDUAL_PATH
        ),
        "team_residual_output_path": str(
            TEAM_RESIDUAL_PATH
        ),
        "source_match_count": int(
            len(expectations)
        ),
        "residual_available_match_count": int(
            len(available)
        ),
        "residual_unavailable_match_count": int(
            len(residuals) - len(available)
        ),
        "team_residual_record_count": int(
            len(team_history)
        ),
        "baseline_names": sorted(
            residuals[
                "baseline_name"
            ].astype(str).unique().tolist()
        ),
        "baseline_versions": sorted(
            residuals[
                "baseline_version"
            ].astype(str).unique().tolist()
        ),
        "feature_specifications": sorted(
            residuals[
                "feature_specification"
            ].astype(str).unique().tolist()
        ),
        "residual_definition_version": "1.0",
        "attack_residual_mean": float(
            attack_summary["mean"]
        ),
        "attack_residual_standard_deviation": float(
            attack_summary[
                "standard_deviation"
            ]
        ),
        "defense_residual_mean": float(
            defense_summary["mean"]
        ),
        "defense_residual_standard_deviation": float(
            defense_summary[
                "standard_deviation"
            ]
        ),
        "residual_identity_validation_pass": True,
        "cross_perspective_validation_pass": True,
        "temporal_validity_pass": True,
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 065 — Historical Residual Repository

## Purpose

Convert the leakage-safe Version 1 expectations produced by
Study 064 into team-perspective attacking and defensive
residuals.

## Residual definitions

For the home team:

- `home_attack_residual`
  = actual home goals - expected home goals
- `home_defense_residual`
  = expected away goals - actual away goals

For the away team:

- `away_attack_residual`
  = actual away goals - expected away goals
- `away_defense_residual`
  = expected home goals - actual home goals

Positive attacking residuals indicate that a team scored more
than expected.

Positive defensive residuals indicate that a team conceded
fewer goals than expected.

## Population

- Source matches:
  {metadata["source_match_count"]}
- Matches with available residuals:
  {metadata["residual_available_match_count"]}
- Warm-up matches without residuals:
  {metadata["residual_unavailable_match_count"]}
- Team-perspective residual records:
  {metadata["team_residual_record_count"]}

## Residual distribution

- Mean team attacking residual:
  {metadata["attack_residual_mean"]:.6f}
- Attacking residual standard deviation:
  {metadata["attack_residual_standard_deviation"]:.6f}
- Mean team defensive residual:
  {metadata["defense_residual_mean"]:.6f}
- Defensive residual standard deviation:
  {metadata["defense_residual_standard_deviation"]:.6f}

The aggregate attacking and defensive means should have equal
magnitudes and opposite signs because each goal residual is
viewed from both teams' perspectives.

## Dynamic Form v1 contract

The future Dynamic Form provider will use:

- Maximum history: 8 eligible matches
- Exponential decay: 0.80
- Only matches strictly before the requested prediction date
- Attack and defense aggregated separately
- No fabricated values when no eligible residual history exists

These aggregation choices are not applied in Study 065. This
study stores the underlying validated residual history.

## Validation

- Study 064 expectation contract: PASS
- Match population preservation: PASS
- Residual availability contract: PASS
- Four residual identities: PASS
- Cross-perspective identities: PASS
- Team-history row derivation: PASS
- Duplicate event/team prevention: PASS
- Temporal validity propagation: PASS
- Distribution summary generation: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def write_outputs(
    residuals: pd.DataFrame,
    audit: pd.DataFrame,
    team_history: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual_output = residuals.copy()
    team_output = team_history.copy()
    audit_output = audit.copy()

    for dataframe in (
        residual_output,
        team_output,
        audit_output,
    ):
        if "date" in dataframe.columns:
            dataframe["date"] = pd.to_datetime(
                dataframe["date"],
                errors="coerce",
                utc=True,
            ).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    for column in (
        "training_start_date",
        "training_end_date",
        "prediction_date",
    ):
        if column in residual_output.columns:
            residual_output[column] = (
                pd.to_datetime(
                    residual_output[column],
                    errors="coerce",
                    utc=True,
                )
                .dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                .replace(
                    {
                        "NaT": "",
                        "nan": "",
                    }
                )
            )

    residual_output.to_csv(
        RESIDUAL_PATH,
        index=False,
    )

    audit_output.to_csv(
        MATCH_AUDIT_PATH,
        index=False,
    )

    team_output.to_csv(
        TEAM_RESIDUAL_PATH,
        index=False,
    )

    distribution_summary.to_csv(
        DISTRIBUTION_PATH,
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


def main() -> None:
    expectations = load_expectations()

    validate_expectation_contract(
        expectations
    )

    residuals = (
        build_residual_repository(
            expectations
        )
    )

    audit = validate_residual_identity(
        residuals
    )

    validate_cross_perspective_identities(
        residuals
    )

    team_history = (
        build_team_residual_history(
            residuals
        )
    )

    distribution_summary = (
        build_distribution_summary(
            residuals=residuals,
            team_history=team_history,
        )
    )

    metadata = build_metadata(
        expectations=expectations,
        residuals=residuals,
        team_history=team_history,
        distribution_summary=(
            distribution_summary
        ),
    )

    write_outputs(
        residuals=residuals,
        audit=audit,
        team_history=team_history,
        distribution_summary=(
            distribution_summary
        ),
        metadata=metadata,
    )

    available = residuals[
        residuals[
            "residual_available"
        ]
    ]

    print(
        "Study 065 — Historical Residual "
        "Repository"
    )
    print("=" * 76)
    print()
    print(
        f"Source matches: {len(expectations)}"
    )
    print(
        "Matches with residuals: "
        f"{len(available)}"
    )
    print(
        "Matches without residuals: "
        f"{len(residuals) - len(available)}"
    )
    print(
        "Team residual records: "
        f"{len(team_history)}"
    )
    print()
    print("Residual Distribution")
    print("-" * 76)
    print(
        distribution_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()
    print("Expectation contract: PASS")
    print("Residual availability contract: PASS")
    print("Residual identity validation: PASS")
    print("Cross-perspective identities: PASS")
    print("Team-history derivation: PASS")
    print("Duplicate event/team prevention: PASS")
    print("Temporal validity propagation: PASS")
    print("Distribution analysis: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()