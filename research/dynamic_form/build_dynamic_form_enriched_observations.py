#build_dynamic_form_enriched_observations

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.dynamic_form.dynamic_form_provider import (
    DEFAULT_DECAY,
    DEFAULT_WINDOW_SIZE,
    DynamicFormRequest,
    HistoricalResidualDynamicFormProvider,
)
from research.modeling.football_feature_registry import (
    get_club_goal_model_feature_spec,
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
    / "study_067_dynamic_form_observation_enrichment"
)

FULL_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "full_squad_observations_with_dynamic_form.csv"
)

MATCHED_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "full_squad_observations_with_complete_dynamic_form.csv"
)

COVERAGE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "dynamic_form_coverage_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "dynamic_form_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


DYNAMIC_FORM_SPECIFICATION = (
    "attack_defense_attack_depth_"
    "rating_prior_dynamic_form"
)

FORM_MODEL_COLUMNS = (
    "home_attack_form",
    "away_attack_form",
    "home_defense_form",
    "away_defense_form",
)

FORM_DIAGNOSTIC_COLUMNS = (
    "attack_form_diff",
    "defense_form_diff",
)

FORM_METADATA_COLUMNS = (
    "home_form_available",
    "away_form_available",
    "both_form_available",
    "home_form_match_count",
    "away_form_match_count",
    "home_form_confidence",
    "away_form_confidence",
    "home_form_earliest_match_date",
    "home_form_latest_match_date",
    "away_form_earliest_match_date",
    "away_form_latest_match_date",
    "home_form_unavailability_reason",
    "away_form_unavailability_reason",
    "form_source",
    "form_window_size",
    "form_decay",
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


def load_source_observations() -> pd.DataFrame:
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

    required_columns = {
        "event_id",
        "date",
        "home_team",
        "home_team_id",
        "away_team",
        "away_team_id",
        "home_score",
        "away_score",
        "home_representation_type",
        "away_representation_type",
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
        "rating_prior_source",
        "rating_prior_available",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Source observations are missing columns: "
            f"{sorted(missing_columns)}"
        )

    conflicting_columns = (
        set(FORM_MODEL_COLUMNS)
        | set(FORM_DIAGNOSTIC_COLUMNS)
        | set(FORM_METADATA_COLUMNS)
    ) & set(dataframe.columns)

    if conflicting_columns:
        raise ValueError(
            "Source observations already contain "
            "Dynamic Form enrichment columns: "
            f"{sorted(conflicting_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    for column in (
        "home_team_id",
        "away_team_id",
    ):
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        ).astype(int)

    dataframe["rating_prior_available"] = (
        parse_boolean(
            dataframe["rating_prior_available"],
            "rating_prior_available",
        )
    )

    if dataframe["event_id"].isna().any():
        raise ValueError(
            "Source observations contain missing "
            "event IDs."
        )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Source observations contain duplicate "
            "event IDs."
        )

    if not dataframe[
        "home_representation_type"
    ].eq("full_squad").all():
        raise ValueError(
            "Unexpected home representation type."
        )

    if not dataframe[
        "away_representation_type"
    ].eq("full_squad").all():
        raise ValueError(
            "Unexpected away representation type."
        )

    if not dataframe[
        "rating_prior_available"
    ].all():
        raise AssertionError(
            "Source observations do not have complete "
            "rating-prior coverage."
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


def unavailability_reason(
    form_available: bool,
    match_count: int,
) -> str:
    if form_available:
        return ""

    if match_count == 0:
        return "no_prior_residual_history"

    return "insufficient_residual_history"


def build_enrichment(
    observations: pd.DataFrame,
    provider: HistoricalResidualDynamicFormProvider,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    for match in observations.itertuples(
        index=False
    ):
        prediction_date = match.date

        home_result = provider.get_form(
            DynamicFormRequest(
                team_id=int(
                    match.home_team_id
                ),
                team_name=str(
                    match.home_team
                ),
                prediction_date=prediction_date,
            )
        )

        away_result = provider.get_form(
            DynamicFormRequest(
                team_id=int(
                    match.away_team_id
                ),
                team_name=str(
                    match.away_team
                ),
                prediction_date=prediction_date,
            )
        )

        both_available = bool(
            home_result.form_available
            and away_result.form_available
        )

        home_attack_form = (
            home_result.attack_form
            if home_result.form_available
            else np.nan
        )

        home_defense_form = (
            home_result.defense_form
            if home_result.form_available
            else np.nan
        )

        away_attack_form = (
            away_result.attack_form
            if away_result.form_available
            else np.nan
        )

        away_defense_form = (
            away_result.defense_form
            if away_result.form_available
            else np.nan
        )

        attack_form_diff = (
            home_attack_form
            - away_attack_form
            if both_available
            else np.nan
        )

        defense_form_diff = (
            home_defense_form
            - away_defense_form
            if both_available
            else np.nan
        )

        output_records.append(
            {
                "event_id": match.event_id,
                "home_attack_form":
                    home_attack_form,
                "away_attack_form":
                    away_attack_form,
                "home_defense_form":
                    home_defense_form,
                "away_defense_form":
                    away_defense_form,
                "attack_form_diff":
                    attack_form_diff,
                "defense_form_diff":
                    defense_form_diff,
                "home_form_available":
                    home_result.form_available,
                "away_form_available":
                    away_result.form_available,
                "both_form_available":
                    both_available,
                "home_form_match_count":
                    home_result.match_count,
                "away_form_match_count":
                    away_result.match_count,
                "home_form_confidence":
                    home_result.confidence,
                "away_form_confidence":
                    away_result.confidence,
                "home_form_earliest_match_date":
                    home_result.earliest_match_date,
                "home_form_latest_match_date":
                    home_result.latest_match_date,
                "away_form_earliest_match_date":
                    away_result.earliest_match_date,
                "away_form_latest_match_date":
                    away_result.latest_match_date,
                "home_form_unavailability_reason":
                    unavailability_reason(
                        form_available=(
                            home_result.form_available
                        ),
                        match_count=(
                            home_result.match_count
                        ),
                    ),
                "away_form_unavailability_reason":
                    unavailability_reason(
                        form_available=(
                            away_result.form_available
                        ),
                        match_count=(
                            away_result.match_count
                        ),
                    ),
                "form_source":
                    provider.provider_name,
                "form_window_size":
                    provider.window_size,
                "form_decay":
                    provider.decay,
            }
        )

        home_temporal_pass = bool(
            (
                not home_result.form_available
            )
            or (
                home_result.latest_match_date
                < prediction_date.date()
            )
        )

        away_temporal_pass = bool(
            (
                not away_result.form_available
            )
            or (
                away_result.latest_match_date
                < prediction_date.date()
            )
        )

        audit_records.append(
            {
                "event_id": match.event_id,
                "date": prediction_date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_form_available":
                    home_result.form_available,
                "away_form_available":
                    away_result.form_available,
                "both_form_available":
                    both_available,
                "home_form_match_count":
                    home_result.match_count,
                "away_form_match_count":
                    away_result.match_count,
                "home_form_confidence":
                    home_result.confidence,
                "away_form_confidence":
                    away_result.confidence,
                "home_latest_match_date":
                    home_result.latest_match_date,
                "away_latest_match_date":
                    away_result.latest_match_date,
                "home_temporal_validity_pass":
                    home_temporal_pass,
                "away_temporal_validity_pass":
                    away_temporal_pass,
                "overall_temporal_validity_pass":
                    (
                        home_temporal_pass
                        and away_temporal_pass
                    ),
            }
        )

    enrichment = pd.DataFrame(
        output_records
    )

    audit = pd.DataFrame(
        audit_records
    )

    enriched = observations.merge(
        enrichment,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    return enriched, audit


def validate_row_preservation(
    source: pd.DataFrame,
    enriched: pd.DataFrame,
) -> None:
    if len(source) != len(enriched):
        raise AssertionError(
            "Dynamic Form enrichment changed the "
            f"observation count: {len(enriched)} vs "
            f"{len(source)}."
        )

    if enriched["event_id"].duplicated().any():
        raise AssertionError(
            "Dynamic Form enrichment produced duplicate "
            "event IDs."
        )

    if set(source["event_id"]) != set(
        enriched["event_id"]
    ):
        raise AssertionError(
            "Dynamic Form enrichment changed the event "
            "population."
        )


def validate_feature_registry_contract(
    enriched: pd.DataFrame,
) -> None:
    specification = (
        get_club_goal_model_feature_spec(
            DYNAMIC_FORM_SPECIFICATION
        )
    )

    missing_columns = (
        set(
            specification.required_columns()
        )
        - set(enriched.columns)
    )

    if missing_columns:
        raise AssertionError(
            "Enriched observations do not satisfy the "
            "Dynamic Form feature specification: "
            f"{sorted(missing_columns)}"
        )

    expected_home_form_features = {
        "home_attack_form",
        "away_defense_form",
    }

    expected_away_form_features = {
        "away_attack_form",
        "home_defense_form",
    }

    if not expected_home_form_features.issubset(
        set(specification.home_features)
    ):
        raise AssertionError(
            "Dynamic Form home-goal feature contract "
            "is invalid."
        )

    if not expected_away_form_features.issubset(
        set(specification.away_features)
    ):
        raise AssertionError(
            "Dynamic Form away-goal feature contract "
            "is invalid."
        )


def validate_availability_contract(
    enriched: pd.DataFrame,
) -> None:
    home_available = enriched[
        "home_form_available"
    ]

    away_available = enriched[
        "away_form_available"
    ]

    both_available = enriched[
        "both_form_available"
    ]

    if not both_available.eq(
        home_available
        & away_available
    ).all():
        raise AssertionError(
            "both_form_available is inconsistent with "
            "home and away availability."
        )

    home_columns = [
        "home_attack_form",
        "home_defense_form",
    ]

    away_columns = [
        "away_attack_form",
        "away_defense_form",
    ]

    if enriched.loc[
        home_available,
        home_columns,
    ].isna().any().any():
        raise AssertionError(
            "Available home form contains missing "
            "values."
        )

    if enriched.loc[
        ~home_available,
        home_columns,
    ].notna().any().any():
        raise AssertionError(
            "Unavailable home form contains fabricated "
            "values."
        )

    if enriched.loc[
        away_available,
        away_columns,
    ].isna().any().any():
        raise AssertionError(
            "Available away form contains missing "
            "values."
        )

    if enriched.loc[
        ~away_available,
        away_columns,
    ].notna().any().any():
        raise AssertionError(
            "Unavailable away form contains fabricated "
            "values."
        )

    if enriched.loc[
        both_available,
        list(FORM_DIAGNOSTIC_COLUMNS),
    ].isna().any().any():
        raise AssertionError(
            "Complete-form observations contain missing "
            "diagnostic differences."
        )

    if enriched.loc[
        ~both_available,
        list(FORM_DIAGNOSTIC_COLUMNS),
    ].notna().any().any():
        raise AssertionError(
            "Incomplete-form observations contain "
            "diagnostic differences."
        )


def validate_difference_arithmetic(
    enriched: pd.DataFrame,
) -> None:
    available = enriched[
        "both_form_available"
    ]

    if not np.allclose(
        enriched.loc[
            available,
            "attack_form_diff",
        ].to_numpy(dtype=float),
        (
            enriched.loc[
                available,
                "home_attack_form",
            ].to_numpy(dtype=float)
            - enriched.loc[
                available,
                "away_attack_form",
            ].to_numpy(dtype=float)
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "attack_form_diff arithmetic is invalid."
        )

    if not np.allclose(
        enriched.loc[
            available,
            "defense_form_diff",
        ].to_numpy(dtype=float),
        (
            enriched.loc[
                available,
                "home_defense_form",
            ].to_numpy(dtype=float)
            - enriched.loc[
                available,
                "away_defense_form",
            ].to_numpy(dtype=float)
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "defense_form_diff arithmetic is invalid."
        )


def validate_provider_metadata(
    enriched: pd.DataFrame,
    provider: HistoricalResidualDynamicFormProvider,
) -> None:
    if not enriched[
        "form_source"
    ].eq(
        provider.provider_name
    ).all():
        raise AssertionError(
            "Unexpected Dynamic Form source."
        )

    if not enriched[
        "form_window_size"
    ].eq(
        provider.window_size
    ).all():
        raise AssertionError(
            "Unexpected Dynamic Form window size."
        )

    if not np.isclose(
        enriched[
            "form_decay"
        ].to_numpy(dtype=float),
        provider.decay,
        atol=0.0,
        rtol=0.0,
    ).all():
        raise AssertionError(
            "Unexpected Dynamic Form decay."
        )

    for column in (
        "home_form_confidence",
        "away_form_confidence",
    ):
        if not enriched[column].between(
            0.0,
            1.0,
            inclusive="both",
        ).all():
            raise AssertionError(
                f"{column} lies outside [0, 1]."
            )

    for column in (
        "home_form_match_count",
        "away_form_match_count",
    ):
        if not enriched[column].between(
            0,
            provider.window_size,
            inclusive="both",
        ).all():
            raise AssertionError(
                f"{column} lies outside the configured "
                "window."
            )


def validate_temporal_integrity(
    audit: pd.DataFrame,
) -> None:
    if not audit[
        "overall_temporal_validity_pass"
    ].all():
        failures = audit[
            ~audit[
                "overall_temporal_validity_pass"
            ]
        ]

        print(
            failures.head(30).to_string(
                index=False
            )
        )

        raise AssertionError(
            "One or more Dynamic Form requests failed "
            "temporal validation."
        )


def build_summary(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    home_available = enriched[
        "home_form_available"
    ]

    away_available = enriched[
        "away_form_available"
    ]

    both_available = enriched[
        "both_form_available"
    ]

    coverage_category = np.select(
        [
            home_available & away_available,
            home_available & ~away_available,
            ~home_available & away_available,
        ],
        [
            "both_available",
            "home_only",
            "away_only",
        ],
        default="neither_available",
    )

    coverage_summary = (
        pd.Series(
            coverage_category,
            name="coverage_category",
        )
        .value_counts()
        .rename_axis("coverage_category")
        .reset_index(name="match_count")
    )

    all_categories = pd.DataFrame(
        {
            "coverage_category": [
                "both_available",
                "home_only",
                "away_only",
                "neither_available",
            ]
        }
    )

    coverage_summary = (
        all_categories.merge(
            coverage_summary,
            on="coverage_category",
            how="left",
        )
        .fillna(
            {
                "match_count": 0,
            }
        )
    )

    coverage_summary[
        "match_count"
    ] = coverage_summary[
        "match_count"
    ].astype(int)

    coverage_summary[
        "coverage_rate"
    ] = (
        coverage_summary[
            "match_count"
        ]
        / len(enriched)
    )

    coverage_summary[
        "total_matches"
    ] = len(enriched)

    return coverage_summary


def build_metadata(
    source: pd.DataFrame,
    enriched: pd.DataFrame,
    matched: pd.DataFrame,
    provider: HistoricalResidualDynamicFormProvider,
) -> dict[str, object]:
    return {
        "study_id": "067",
        "study_name": (
            "Dynamic Form Observation Enrichment"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_path": str(INPUT_PATH),
        "full_output_path": str(
            FULL_OUTPUT_PATH
        ),
        "matched_output_path": str(
            MATCHED_OUTPUT_PATH
        ),
        "source_observation_count": int(
            len(source)
        ),
        "enriched_observation_count": int(
            len(enriched)
        ),
        "complete_form_observation_count": int(
            len(matched)
        ),
        "complete_form_coverage_rate": float(
            len(matched) / len(enriched)
        ),
        "provider_name":
            provider.provider_name,
        "window_size":
            provider.window_size,
        "decay":
            provider.decay,
        "candidate_feature_specification":
            DYNAMIC_FORM_SPECIFICATION,
        "home_form_available_count": int(
            enriched[
                "home_form_available"
            ].sum()
        ),
        "away_form_available_count": int(
            enriched[
                "away_form_available"
            ].sum()
        ),
        "mean_home_form_confidence": float(
            enriched[
                "home_form_confidence"
            ].mean()
        ),
        "mean_away_form_confidence": float(
            enriched[
                "away_form_confidence"
            ].mean()
        ),
    }


def write_report(
    metadata: dict[str, object],
    summary: pd.DataFrame,
) -> None:
    coverage_lines = "\n".join(
        (
            f"- {row.coverage_category}: "
            f"{int(row.match_count)} "
            f"({float(row.coverage_rate):.2%})"
        )
        for row in summary.itertuples(
            index=False
        )
    )

    report = f"""# Study 067 — Dynamic Form Observation Enrichment

## Purpose

Add leakage-safe Dynamic Form values to the validated
ClubElo-enriched full-squad observation dataset.

## Dynamic Form definition

- Maximum residual history: {metadata["window_size"]} matches
- Exponential decay: {metadata["decay"]:.2f}
- Only residual matches strictly before the prediction date
- Attacking and defensive form aggregated separately

## Modeling fields

### Home-goal model contribution

- `home_attack_form`
- `away_defense_form`

### Away-goal model contribution

- `away_attack_form`
- `home_defense_form`

The difference fields are retained for diagnostics but are not
the primary model representation.

## Population

- Source observations:
  {metadata["source_observation_count"]}
- Enriched observations:
  {metadata["enriched_observation_count"]}
- Complete-form observations:
  {metadata["complete_form_observation_count"]}
- Complete-form coverage:
  {metadata["complete_form_coverage_rate"]:.2%}

## Coverage

{coverage_lines}

## Candidate feature specification

`{metadata["candidate_feature_specification"]}`

The current Version 1 baseline remains unchanged until Dynamic
Form survives the incremental benchmark.

## Validation

- Source row preservation: PASS
- Event-population preservation: PASS
- Provider identity integration: PASS
- Strict prior-date form history: PASS
- Form availability contract: PASS
- No fabricated unavailable values: PASS
- Eight-match history cap: PASS
- Exponential-decay metadata: PASS
- Diagnostic-difference arithmetic: PASS
- Feature-registry integration: PASS
- Matched benchmark population generation: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def write_outputs(
    enriched: pd.DataFrame,
    matched: pd.DataFrame,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_output = enriched.copy()
    matched_output = matched.copy()
    audit_output = audit.copy()

    date_columns = (
        "date",
        "home_form_earliest_match_date",
        "home_form_latest_match_date",
        "away_form_earliest_match_date",
        "away_form_latest_match_date",
    )

    for dataframe in (
        full_output,
        matched_output,
    ):
        for column in date_columns:
            if column not in dataframe.columns:
                continue

            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
                utc=True,
            ).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

            dataframe[column] = (
                dataframe[column]
                .replace(
                    {
                        "NaT": "",
                        "nan": "",
                    }
                )
            )

    for column in (
        "date",
        "home_latest_match_date",
        "away_latest_match_date",
    ):
        if column in audit_output.columns:
            audit_output[column] = (
                pd.to_datetime(
                    audit_output[column],
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

    full_output.to_csv(
        FULL_OUTPUT_PATH,
        index=False,
    )

    matched_output.to_csv(
        MATCHED_OUTPUT_PATH,
        index=False,
    )

    audit_output.to_csv(
        COVERAGE_AUDIT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
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
        metadata=metadata,
        summary=summary,
    )


def main() -> None:
    source = load_source_observations()

    provider = (
        HistoricalResidualDynamicFormProvider(
            window_size=DEFAULT_WINDOW_SIZE,
            decay=DEFAULT_DECAY,
        )
    )

    enriched, audit = build_enrichment(
        observations=source,
        provider=provider,
    )

    validate_row_preservation(
        source=source,
        enriched=enriched,
    )

    validate_feature_registry_contract(
        enriched
    )

    validate_availability_contract(
        enriched
    )

    validate_difference_arithmetic(
        enriched
    )

    validate_provider_metadata(
        enriched=enriched,
        provider=provider,
    )

    validate_temporal_integrity(
        audit
    )

    matched = (
        enriched[
            enriched[
                "both_form_available"
            ]
        ]
        .copy()
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    if matched.empty:
        raise AssertionError(
            "No complete Dynamic Form benchmark "
            "population was produced."
        )

    if matched[
        list(FORM_MODEL_COLUMNS)
    ].isna().any().any():
        raise AssertionError(
            "Matched Dynamic Form population contains "
            "missing model values."
        )

    summary = build_summary(
        enriched
    )

    metadata = build_metadata(
        source=source,
        enriched=enriched,
        matched=matched,
        provider=provider,
    )

    write_outputs(
        enriched=enriched,
        matched=matched,
        audit=audit,
        summary=summary,
        metadata=metadata,
    )

    print(
        "Study 067 — Dynamic Form Observation "
        "Enrichment"
    )
    print("=" * 76)
    print()
    print(
        f"Source observations: {len(source)}"
    )
    print(
        f"Enriched observations: {len(enriched)}"
    )
    print(
        "Complete-form observations: "
        f"{len(matched)}"
    )
    print(
        "Complete-form coverage: "
        f"{len(matched) / len(enriched):.2%}"
    )
    print()
    print("Coverage Summary")
    print("-" * 76)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()
    print("Source row preservation: PASS")
    print("Event-population preservation: PASS")
    print("Provider integration: PASS")
    print("Strict prior-date filtering: PASS")
    print("Form availability contract: PASS")
    print("Diagnostic arithmetic: PASS")
    print("Feature-registry integration: PASS")
    print("Matched benchmark population: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()