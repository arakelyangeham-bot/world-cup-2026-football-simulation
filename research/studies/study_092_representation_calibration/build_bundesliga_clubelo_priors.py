#build_bundesliga_clubelo_priors

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)
from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)
from research.studies.study_083_bundesliga_production_replay.run_bundesliga_production_replay import (
    load_replay_population,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c2"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "bundesliga_match_rating_priors.csv"
)

AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "bundesliga_match_rating_prior_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092c1c_metadata.json"
)

EXPECTED_MATCH_COUNT = 306


OUTPUT_COLUMNS = (
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
)


def resolve_lookup_name(
    production_club: str,
) -> str:
    try:
        return BUNDESLIGA_CLUBELO_NAME_OVERRIDES[
            production_club
        ]
    except KeyError as error:
        raise KeyError(
            "Bundesliga ClubElo override is missing for "
            f"{production_club!r}."
        ) from error


def build_rating_priors(
    fixtures: pd.DataFrame,
    repository: ClubEloRepository,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for fixture in fixtures.itertuples(index=False):
        match_date = pd.Timestamp(
            fixture.date
        ).date()

        home_lookup_name = resolve_lookup_name(
            str(fixture.home_team)
        )

        away_lookup_name = resolve_lookup_name(
            str(fixture.away_team)
        )

        try:
            home_result = repository.resolve_rating(
                club_name=home_lookup_name,
                prediction_date=match_date,
                refresh=False,
            )

            away_result = repository.resolve_rating(
                club_name=away_lookup_name,
                prediction_date=match_date,
                refresh=False,
            )

            rating_difference = (
                home_result.rating
                - away_result.rating
            )

            rows.append(
                {
                    "event_id": fixture.event_id,
                    "date": fixture.date,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "home_rating_prior":
                        home_result.rating,
                    "away_rating_prior":
                        away_result.rating,
                    "rating_prior_diff":
                        rating_difference,
                    "rating_prior_source":
                        "clubelo",
                    "rating_prior_available":
                        True,
                    "home_rating_effective_from":
                        home_result.effective_from,
                    "home_rating_effective_to":
                        home_result.effective_to,
                    "away_rating_effective_from":
                        away_result.effective_from,
                    "away_rating_effective_to":
                        away_result.effective_to,
                }
            )

            audit_rows.append(
                {
                    "event_id": fixture.event_id,
                    "date": fixture.date,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "home_lookup_name":
                        home_lookup_name,
                    "away_lookup_name":
                        away_lookup_name,
                    "home_resolved_club":
                        home_result.resolved_club,
                    "away_resolved_club":
                        away_result.resolved_club,
                    "home_temporal_validity_pass":
                        home_result.temporal_validity_pass,
                    "away_temporal_validity_pass":
                        away_result.temporal_validity_pass,
                    "rating_difference_validation_pass":
                        np.isclose(
                            rating_difference,
                            (
                                home_result.rating
                                - away_result.rating
                            ),
                            atol=1e-12,
                            rtol=0.0,
                        ),
                    "resolution_status":
                        "PASS",
                    "resolution_error":
                        "",
                }
            )

        except Exception as error:
            audit_rows.append(
                {
                    "event_id": fixture.event_id,
                    "date": fixture.date,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "home_lookup_name":
                        home_lookup_name,
                    "away_lookup_name":
                        away_lookup_name,
                    "home_resolved_club":
                        pd.NA,
                    "away_resolved_club":
                        pd.NA,
                    "home_temporal_validity_pass":
                        False,
                    "away_temporal_validity_pass":
                        False,
                    "rating_difference_validation_pass":
                        False,
                    "resolution_status":
                        "FAIL",
                    "resolution_error":
                        f"{type(error).__name__}: {error}",
                }
            )

    priors = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    audit = pd.DataFrame(
        audit_rows
    )

    return priors, audit


def validate_fixture_population(
    fixtures: pd.DataFrame,
) -> None:
    if len(fixtures) != EXPECTED_MATCH_COUNT:
        raise AssertionError(
            "Unexpected fixture count. "
            f"Expected {EXPECTED_MATCH_COUNT}, "
            f"received {len(fixtures)}."
        )

    fixture_clubs = set(
        fixtures["home_team"].astype(str)
    ) | set(
        fixtures["away_team"].astype(str)
    )

    override_clubs = set(
        BUNDESLIGA_CLUBELO_NAME_OVERRIDES
    )

    missing_overrides = sorted(
        fixture_clubs - override_clubs,
        key=str.casefold,
    )

    extra_overrides = sorted(
        override_clubs - fixture_clubs,
        key=str.casefold,
    )

    if missing_overrides:
        raise ValueError(
            "Fixture clubs lack ClubElo overrides: "
            f"{missing_overrides}"
        )

    if extra_overrides:
        raise ValueError(
            "ClubElo override table contains clubs outside "
            "the fixture population: "
            f"{extra_overrides}"
        )


def validate_outputs(
    fixtures: pd.DataFrame,
    priors: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    failures = audit.loc[
        audit["resolution_status"].ne("PASS")
    ]

    if not failures.empty:
        print()
        print("ClubElo resolution failures")
        print("-" * 88)
        print(
            failures[
                [
                    "event_id",
                    "date",
                    "home_team",
                    "away_team",
                    "home_lookup_name",
                    "away_lookup_name",
                    "resolution_error",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

        raise AssertionError(
            "One or more Bundesliga ClubElo lookups failed."
        )

    if len(priors) != len(fixtures):
        raise AssertionError(
            "Rating-prior row count differs from fixture "
            f"population: priors={len(priors)}, "
            f"fixtures={len(fixtures)}."
        )

    if priors["event_id"].duplicated().any():
        raise AssertionError(
            "Rating-prior output contains duplicate event IDs."
        )

    if set(priors["event_id"]) != set(
        fixtures["event_id"]
    ):
        raise AssertionError(
            "Rating-prior output does not preserve the exact "
            "fixture event population."
        )

    numeric_columns = [
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
    ]

    numeric = priors[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise AssertionError(
            "Rating-prior output contains non-finite values."
        )

    expected_difference = (
        priors["home_rating_prior"]
        - priors["away_rating_prior"]
    )

    if not np.allclose(
        priors["rating_prior_diff"].to_numpy(
            dtype=float
        ),
        expected_difference.to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "rating_prior_diff arithmetic validation failed."
        )

    if set(
        priors["rating_prior_source"]
    ) != {"clubelo"}:
        raise AssertionError(
            "Unexpected rating-prior source values."
        )

    if not priors[
        "rating_prior_available"
    ].astype(bool).all():
        raise AssertionError(
            "One or more rating priors are unavailable."
        )

    match_dates = pd.to_datetime(
        priors["date"],
        errors="raise",
        utc=True,
    ).dt.date

    home_from = pd.to_datetime(
        priors["home_rating_effective_from"],
        errors="raise",
    ).dt.date

    home_to = pd.to_datetime(
        priors["home_rating_effective_to"],
        errors="raise",
    ).dt.date

    away_from = pd.to_datetime(
        priors["away_rating_effective_from"],
        errors="raise",
    ).dt.date

    away_to = pd.to_datetime(
        priors["away_rating_effective_to"],
        errors="raise",
    ).dt.date

    home_temporal_pass = (
        home_from.le(match_dates)
        & match_dates.le(home_to)
    )

    away_temporal_pass = (
        away_from.le(match_dates)
        & match_dates.le(away_to)
    )

    if not home_temporal_pass.all():
        raise AssertionError(
            "One or more home ClubElo intervals are not "
            "valid on the fixture date."
        )

    if not away_temporal_pass.all():
        raise AssertionError(
            "One or more away ClubElo intervals are not "
            "valid on the fixture date."
        )

    if not audit[
        "home_temporal_validity_pass"
    ].all():
        raise AssertionError(
            "Home temporal-validity audit failed."
        )

    if not audit[
        "away_temporal_validity_pass"
    ].all():
        raise AssertionError(
            "Away temporal-validity audit failed."
        )

    if not audit[
        "rating_difference_validation_pass"
    ].all():
        raise AssertionError(
            "Rating-difference audit failed."
        )


def format_output_dates(
    priors: pd.DataFrame,
) -> pd.DataFrame:
    output = priors.copy()

    output["date"] = pd.to_datetime(
        output["date"],
        errors="raise",
        utc=True,
    ).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    for column in (
        "home_rating_effective_from",
        "home_rating_effective_to",
        "away_rating_effective_from",
        "away_rating_effective_to",
    ):
        output[column] = pd.to_datetime(
            output[column],
            errors="raise",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    return output


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 092C1C — BUNDESLIGA "
        "FIXTURE-DATE CLUBELO PRIORS"
    )
    print("=" * 88)

    fixtures = load_replay_population()

    validate_fixture_population(
        fixtures
    )

    repository = ClubEloRepository(
        cache_directory=CLUBELO_CACHE_DIRECTORY
    )

    priors, audit = build_rating_priors(
        fixtures=fixtures,
        repository=repository,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write the audit before validation so failures remain inspectable.
    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    validate_outputs(
        fixtures=fixtures,
        priors=priors,
        audit=audit,
    )

    output = format_output_dates(
        priors
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    metadata = {
        "study_id": "092C1C",
        "study_name": (
            "Bundesliga Fixture-Date ClubElo Priors"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "fixture_count": len(fixtures),
        "prior_row_count": len(output),
        "unique_event_count": int(
            output["event_id"].nunique()
        ),
        "club_count": len(
            set(fixtures["home_team"])
            | set(fixtures["away_team"])
        ),
        "rating_prior_source": "clubelo",
        "complete_rating_coverage": True,
        "temporal_validity_pass": True,
        "rating_difference_validation_pass": True,
        "cache_directory": str(
            CLUBELO_CACHE_DIRECTORY.relative_to(
                PROJECT_ROOT
            )
        ),
        "methodological_role": (
            "Representation-invariant fixture-date prior "
            "used identically across all Study 092C branches."
        ),
        "outputs": [
            OUTPUT_PATH.name,
            AUDIT_PATH.name,
            METADATA_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Generation summary")
    print("-" * 88)
    print(
        f"  Fixtures loaded: {len(fixtures)}"
    )
    print(
        f"  Rating-prior rows: {len(output)}"
    )
    print(
        "  Unique events: "
        f"{output['event_id'].nunique()}"
    )
    print(
        "  Mean home rating: "
        f"{output['home_rating_prior'].mean():.6f}"
    )
    print(
        "  Mean away rating: "
        f"{output['away_rating_prior'].mean():.6f}"
    )
    print(
        "  Minimum rating difference: "
        f"{output['rating_prior_diff'].min():.6f}"
    )
    print(
        "  Maximum rating difference: "
        f"{output['rating_prior_diff'].max():.6f}"
    )

    print()
    print("Validation summary")
    print("  Fixture population preserved: PASS")
    print("  Club override coverage: PASS")
    print("  ClubElo cache resolution: PASS")
    print("  Complete event coverage: PASS")
    print("  Unique event IDs: PASS")
    print("  Finite rating values: PASS")
    print("  Rating-difference arithmetic: PASS")
    print("  ClubElo provenance: PASS")
    print("  Historical temporal validity: PASS")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()