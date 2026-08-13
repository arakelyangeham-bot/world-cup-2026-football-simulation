#build_clubelo_enriched_observations

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

RATING_PRIOR_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_059_rating_prior_provider"
    / "match_rating_priors.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
)

OUTPUT_OBSERVATION_PATH = (
    OUTPUT_DIRECTORY
    / "full_squad_observations_with_clubelo.csv"
)

MERGE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_enrichment_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_enrichment_summary.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_enrichment_report.md"
)


SOURCE_REQUIRED_COLUMNS = {
    "event_id",
    "date",
    "home_team",
    "away_team",
    "home_representation_type",
    "away_representation_type",
}

PRIOR_REQUIRED_COLUMNS = {
    "event_id",
    "date",
    "home_team",
    "away_team",
    "home_rating_prior",
    "away_rating_prior",
    "rating_prior_diff",
    "rating_prior_source",
    "rating_prior_available",
    "home_rating_effective_from",
    "home_rating_effective_to",
    "away_rating_effective_from",
    "away_rating_effective_to",
}

ENRICHMENT_COLUMNS = (
    "home_rating_prior",
    "away_rating_prior",
    "rating_prior_diff",
    "rating_prior_source",
    "rating_prior_available",
    "home_rating_effective_from",
    "home_rating_effective_to",
    "away_rating_effective_from",
    "away_rating_effective_to",
)


def load_csv(
    path: Path,
    required_columns: set[str],
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"{label} is empty: {path}"
        )

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe["event_id"].isna().any():
        raise ValueError(
            f"{label} contains missing event IDs."
        )

    if dataframe["event_id"].duplicated().any():
        duplicate_ids = (
            dataframe.loc[
                dataframe["event_id"].duplicated(
                    keep=False
                ),
                "event_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{label} contains duplicate event IDs: "
            f"{duplicate_ids[:20]}"
        )

    return dataframe.copy()


def normalize_dates(
    observations: pd.DataFrame,
    priors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = observations.copy()
    priors = priors.copy()

    observations["date"] = pd.to_datetime(
        observations["date"],
        errors="raise",
        utc=True,
    )

    priors["date"] = pd.to_datetime(
        priors["date"],
        errors="raise",
        utc=True,
    )

    for column in (
        "home_rating_effective_from",
        "home_rating_effective_to",
        "away_rating_effective_from",
        "away_rating_effective_to",
    ):
        priors[column] = pd.to_datetime(
            priors[column],
            errors="raise",
        ).dt.date

    return observations, priors


def validate_source_observations(
    observations: pd.DataFrame,
) -> None:
    """
    Validate the player-derived source observations.

    The original observation schema may already contain
    unpopulated rating-prior placeholder columns. These are
    permitted, but populated historical rating values are not,
    because this study must not silently overwrite an existing
    prior.
    """
    if not observations[
        "home_representation_type"
    ].eq("full_squad").all():
        raise ValueError(
            "Source observations contain unexpected "
            "home representation types."
        )

    if not observations[
        "away_representation_type"
    ].eq("full_squad").all():
        raise ValueError(
            "Source observations contain unexpected "
            "away representation types."
        )

    existing_columns = [
        column
        for column in ENRICHMENT_COLUMNS
        if column in observations.columns
    ]

    for column in existing_columns:
        series = observations[column]

        if column == "rating_prior_available":
            normalized = (
                series.astype(str)
                .str.strip()
                .str.lower()
            )

            populated = ~normalized.isin(
                {
                    "",
                    "nan",
                    "none",
                    "<na>",
                    "false",
                    "0",
                }
            )

        elif column == "rating_prior_source":
            normalized = (
                series.fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            populated = ~normalized.isin(
                {
                    "",
                    "none",
                    "nan",
                    "<na>",
                    "unavailable",
                }
            )

        else:
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            populated = numeric.notna()

        if populated.any():
            raise ValueError(
                "Source observations contain populated "
                "rating-prior values that this study "
                "would overwrite. "
                f"Column: {column!r}; populated rows: "
                f"{int(populated.sum())}"
            )

def remove_rating_prior_placeholders(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove predeclared but unpopulated rating-prior schema
    columns before attaching validated ClubElo values.

    The source dataframe itself remains unchanged.
    """
    placeholder_columns = [
        column
        for column in ENRICHMENT_COLUMNS
        if column in observations.columns
    ]

    return observations.drop(
        columns=placeholder_columns
    ).copy()

def validate_rating_priors(
    priors: pd.DataFrame,
) -> None:
    numeric_columns = (
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
    )

    for column in numeric_columns:
        priors[column] = pd.to_numeric(
            priors[column],
            errors="raise",
        )

    if priors[
        list(numeric_columns)
    ].isna().any().any():
        raise ValueError(
            "Rating-prior dataset contains missing "
            "numeric values."
        )

    if not np.isfinite(
        priors[
            list(numeric_columns)
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Rating-prior dataset contains non-finite "
            "numeric values."
        )

    derived_difference = (
        priors["home_rating_prior"]
        - priors["away_rating_prior"]
    )

    if not np.allclose(
        priors["rating_prior_diff"].to_numpy(
            dtype=float
        ),
        derived_difference.to_numpy(dtype=float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise AssertionError(
            "rating_prior_diff does not equal home "
            "rating minus away rating."
        )

    source_values = set(
        priors["rating_prior_source"]
        .dropna()
        .astype(str)
    )

    if source_values != {"clubelo"}:
        raise AssertionError(
            "Unexpected rating-prior sources: "
            f"{sorted(source_values)}"
        )

    available = (
        priors["rating_prior_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if available.isna().any():
        raise ValueError(
            "rating_prior_available contains "
            "unrecognized values."
        )

    priors["rating_prior_available"] = available

    if not priors[
        "rating_prior_available"
    ].all():
        raise AssertionError(
            "One or more historical rating priors are "
            "marked unavailable."
        )


def build_population_audit(
    observations: pd.DataFrame,
    priors: pd.DataFrame,
) -> pd.DataFrame:
    observation_population = observations[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
        ]
    ].rename(
        columns={
            "date": "observation_date",
            "home_team": "observation_home_team",
            "away_team": "observation_away_team",
        }
    )

    prior_population = priors[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
        ]
    ].rename(
        columns={
            "date": "prior_date",
            "home_team": "prior_home_team",
            "away_team": "prior_away_team",
        }
    )

    audit = observation_population.merge(
        prior_population,
        on="event_id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    audit["event_population_match"] = (
        audit["_merge"].eq("both")
    )

    audit["date_match"] = (
        audit["observation_date"].dt.date
        == audit["prior_date"].dt.date
    )

    audit["home_team_match"] = (
        audit["observation_home_team"]
        .fillna("<missing>")
        .astype(str)
        .eq(
            audit["prior_home_team"]
            .fillna("<missing>")
            .astype(str)
        )
    )

    audit["away_team_match"] = (
        audit["observation_away_team"]
        .fillna("<missing>")
        .astype(str)
        .eq(
            audit["prior_away_team"]
            .fillna("<missing>")
            .astype(str)
        )
    )

    audit["population_validation_pass"] = (
        audit["event_population_match"]
        & audit["date_match"]
        & audit["home_team_match"]
        & audit["away_team_match"]
    )

    return audit


def validate_population_audit(
    audit: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(audit) != expected_rows:
        raise AssertionError(
            "Merged event population has an unexpected "
            f"row count: {len(audit)} vs "
            f"{expected_rows}."
        )

    failures = audit[
        ~audit["population_validation_pass"]
    ]

    if not failures.empty:
        columns = [
            "event_id",
            "_merge",
            "observation_date",
            "prior_date",
            "observation_home_team",
            "prior_home_team",
            "observation_away_team",
            "prior_away_team",
            "event_population_match",
            "date_match",
            "home_team_match",
            "away_team_match",
        ]

        print(
            failures[columns]
            .head(30)
            .to_string(index=False)
        )

        raise AssertionError(
            "Observation and rating-prior populations "
            "do not match exactly."
        )


def build_enriched_observations(
    observations: pd.DataFrame,
    priors: pd.DataFrame,
) -> pd.DataFrame:
    prior_columns = [
        "event_id",
        *ENRICHMENT_COLUMNS,
    ]

    enriched = observations.merge(
        priors[prior_columns],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    if len(enriched) != len(observations):
        raise AssertionError(
            "Enrichment changed the observation row "
            "count."
        )

    if enriched["event_id"].duplicated().any():
        raise AssertionError(
            "Enrichment created duplicate event IDs."
        )

    for column in ENRICHMENT_COLUMNS:
        if enriched[column].isna().any():
            raise AssertionError(
                "Enriched observations contain missing "
                f"values in {column!r}."
            )

    return enriched


def validate_temporal_provenance(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    audit = enriched[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "home_rating_effective_from",
            "home_rating_effective_to",
            "away_rating_effective_from",
            "away_rating_effective_to",
        ]
    ].copy()

    audit["match_calendar_date"] = (
        audit["date"].dt.date
    )

    audit["home_temporal_validity_pass"] = (
        (
            audit["home_rating_effective_from"]
            <= audit["match_calendar_date"]
        )
        & (
            audit["match_calendar_date"]
            <= audit["home_rating_effective_to"]
        )
    )

    audit["away_temporal_validity_pass"] = (
        (
            audit["away_rating_effective_from"]
            <= audit["match_calendar_date"]
        )
        & (
            audit["match_calendar_date"]
            <= audit["away_rating_effective_to"]
        )
    )

    audit["temporal_validity_pass"] = (
        audit["home_temporal_validity_pass"]
        & audit["away_temporal_validity_pass"]
    )

    if not audit[
        "temporal_validity_pass"
    ].all():
        failures = audit[
            ~audit["temporal_validity_pass"]
        ]

        print(
            failures.head(30).to_string(
                index=False
            )
        )

        raise AssertionError(
            "One or more enriched observations failed "
            "ClubElo temporal validation."
        )

    return audit


def build_summary(
    observations: pd.DataFrame,
    priors: pd.DataFrame,
    enriched: pd.DataFrame,
    population_audit: pd.DataFrame,
    temporal_audit: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study": (
            "study_060_clubelo_enriched_observations"
        ),
        "source_observation_path": str(
            SOURCE_OBSERVATION_PATH
        ),
        "rating_prior_path": str(
            RATING_PRIOR_PATH
        ),
        "output_observation_path": str(
            OUTPUT_OBSERVATION_PATH
        ),
        "source_observation_rows": int(
            len(observations)
        ),
        "rating_prior_rows": int(
            len(priors)
        ),
        "enriched_observation_rows": int(
            len(enriched)
        ),
        "unique_events": int(
            enriched["event_id"].nunique()
        ),
        "unique_home_teams": int(
            enriched["home_team"].nunique()
        ),
        "unique_away_teams": int(
            enriched["away_team"].nunique()
        ),
        "rating_prior_source_values": sorted(
            enriched["rating_prior_source"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "population_validation_pass": bool(
            population_audit[
                "population_validation_pass"
            ].all()
        ),
        "temporal_validity_pass": bool(
            temporal_audit[
                "temporal_validity_pass"
            ].all()
        ),
        "rating_difference_validation_pass": bool(
            np.allclose(
                enriched[
                    "rating_prior_diff"
                ].to_numpy(dtype=float),
                (
                    enriched[
                        "home_rating_prior"
                    ].to_numpy(dtype=float)
                    - enriched[
                        "away_rating_prior"
                    ].to_numpy(dtype=float)
                ),
                atol=1e-10,
                rtol=0.0,
            )
        ),
        "rating_prior_summary": {
            "home_mean": float(
                enriched[
                    "home_rating_prior"
                ].mean()
            ),
            "away_mean": float(
                enriched[
                    "away_rating_prior"
                ].mean()
            ),
            "difference_mean": float(
                enriched[
                    "rating_prior_diff"
                ].mean()
            ),
            "difference_min": float(
                enriched[
                    "rating_prior_diff"
                ].min()
            ),
            "difference_max": float(
                enriched[
                    "rating_prior_diff"
                ].max()
            ),
        },
    }


def write_report(
    summary: dict[str, object],
) -> None:
    rating_summary = summary[
        "rating_prior_summary"
    ]

    report = f"""# Study 060 — ClubElo-Enriched Observations

## Purpose

Augment the validated full-squad club observation dataset with
temporally valid historical ClubElo rating priors.

## Inputs

- Source observations: `{SOURCE_OBSERVATION_PATH}`
- Match rating priors: `{RATING_PRIOR_PATH}`

## Output

- Enriched observations: `{OUTPUT_OBSERVATION_PATH}`

## Population

- Source observation rows: {summary["source_observation_rows"]}
- Rating-prior rows: {summary["rating_prior_rows"]}
- Enriched observation rows: {summary["enriched_observation_rows"]}
- Unique events: {summary["unique_events"]}

## Rating-prior summary

- Mean home rating: {rating_summary["home_mean"]:.6f}
- Mean away rating: {rating_summary["away_mean"]:.6f}
- Mean rating difference: {rating_summary["difference_mean"]:.6f}
- Minimum rating difference: {rating_summary["difference_min"]:.6f}
- Maximum rating difference: {rating_summary["difference_max"]:.6f}

## Validation

- Event population match: PASS
- Match-date agreement: PASS
- Team-name agreement: PASS
- Complete rating coverage: PASS
- Rating-difference arithmetic: PASS
- ClubElo provenance: PASS
- Historical temporal validity: PASS
- Source row preservation: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    observations = load_csv(
        path=SOURCE_OBSERVATION_PATH,
        required_columns=SOURCE_REQUIRED_COLUMNS,
        label="Source observation dataset",
    )

    priors = load_csv(
        path=RATING_PRIOR_PATH,
        required_columns=PRIOR_REQUIRED_COLUMNS,
        label="Rating-prior dataset",
    )

    observations, priors = normalize_dates(
        observations=observations,
        priors=priors,
    )

    validate_source_observations(
        observations
    )

    source_observations = observations.copy()

    observations = (
        remove_rating_prior_placeholders(
            observations
        )
    )

    validate_rating_priors(
        priors
    )

    population_audit = build_population_audit(
        observations=observations,
        priors=priors,
    )

    validate_population_audit(
        audit=population_audit,
        expected_rows=len(observations),
    )

    enriched = build_enriched_observations(
        observations=observations,
        priors=priors,
    )

    temporal_audit = validate_temporal_provenance(
        enriched
    )

    summary = build_summary(
        observations=source_observations,
        priors=priors,
        enriched=enriched,
        population_audit=population_audit,
        temporal_audit=temporal_audit,
    )

    if not summary[
        "rating_difference_validation_pass"
    ]:
        raise AssertionError(
            "Enriched rating-prior differences failed "
            "arithmetic validation."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched_output = enriched.copy()

    enriched_output["date"] = (
        enriched_output["date"]
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    enriched_output.to_csv(
        OUTPUT_OBSERVATION_PATH,
        index=False,
    )

    output_audit = population_audit.merge(
        temporal_audit[
            [
                "event_id",
                "home_temporal_validity_pass",
                "away_temporal_validity_pass",
                "temporal_validity_pass",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    output_audit.to_csv(
        MERGE_AUDIT_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(summary)

    rating_summary = summary[
        "rating_prior_summary"
    ]

    print(
        "Study 060 — ClubElo Observation Enrichment"
    )
    print("=" * 76)
    print()
    print(
        "Source observation rows: "
        f"{len(observations)}"
    )
    print(
        "Rating-prior rows: "
        f"{len(priors)}"
    )
    print(
        "Enriched observation rows: "
        f"{len(enriched)}"
    )
    print(
        "Unique events: "
        f"{enriched['event_id'].nunique()}"
    )
    print()
    print("Rating Prior Summary")
    print("-" * 76)
    print(
        "Mean home rating: "
        f"{rating_summary['home_mean']:.6f}"
    )
    print(
        "Mean away rating: "
        f"{rating_summary['away_mean']:.6f}"
    )
    print(
        "Mean rating difference: "
        f"{rating_summary['difference_mean']:.6f}"
    )
    print(
        "Minimum rating difference: "
        f"{rating_summary['difference_min']:.6f}"
    )
    print(
        "Maximum rating difference: "
        f"{rating_summary['difference_max']:.6f}"
    )
    print()
    print("Event population match: PASS")
    print("Match-date agreement: PASS")
    print("Team-name agreement: PASS")
    print("Complete rating coverage: PASS")
    print("Rating-difference arithmetic: PASS")
    print("ClubElo provenance: PASS")
    print("Historical temporal validity: PASS")
    print("Source row preservation: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()