#validate_rating_prior_provider

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)
from research.rating_priors.rating_prior_provider import (
    ClubEloRatingPriorProvider,
    RatingPriorRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "clubelo_histories"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_059_rating_prior_provider"
)


def load_match_requests() -> pd.DataFrame:
    if not OBSERVATION_PATH.exists():
        raise FileNotFoundError(
            "Observation dataset does not exist: "
            f"{OBSERVATION_PATH}"
        )

    dataframe = pd.read_csv(
        OBSERVATION_PATH,
        usecols=[
            "event_id",
            "date",
            "home_team",
            "away_team",
        ],
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Observation dataset is empty."
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Observation dataset contains duplicate "
            "event IDs."
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


def build_provider_audit(
    matches: pd.DataFrame,
    provider: ClubEloRatingPriorProvider,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for match in matches.itertuples(
        index=False
    ):
        for venue_role, team_name in (
            (
                "home",
                str(match.home_team),
            ),
            (
                "away",
                str(match.away_team),
            ),
        ):
            record: dict[str, object] = {
                "event_id": match.event_id,
                "match_date": (
                    match.date.date().isoformat()
                ),
                "venue_role": venue_role,
                "requested_team_name":
                    team_name,
                "canonical_team_name": "",
                "source_team_name": "",
                "rating": np.nan,
                "source": "",
                "effective_from": "",
                "effective_to": "",
                "source_rank": np.nan,
                "source_country": "",
                "source_level": np.nan,
                "rating_available": False,
                "temporal_validity_pass": False,
                "error_type": "",
                "error_message": "",
                "overall_validation_pass": False,
            }

            try:
                result = (
                    provider.get_rating_prior(
                        RatingPriorRequest(
                            team_name=team_name,
                            prediction_date=(
                                match.date
                            ),
                        )
                    )
                )

                temporal_pass = (
                    result.effective_from
                    <= result.prediction_date
                    <= result.effective_to
                )

                rating_pass = (
                    result.rating_available
                    and np.isfinite(
                        result.rating
                    )
                )

                source_pass = (
                    result.source
                    == provider.source_name
                )

                overall_pass = (
                    temporal_pass
                    and rating_pass
                    and source_pass
                )

                record.update(
                    {
                        "canonical_team_name":
                            result.canonical_team_name,
                        "source_team_name":
                            result.source_team_name,
                        "rating":
                            result.rating,
                        "source":
                            result.source,
                        "effective_from":
                            result.effective_from.isoformat(),
                        "effective_to":
                            result.effective_to.isoformat(),
                        "source_rank":
                            result.source_rank,
                        "source_country":
                            result.source_country,
                        "source_level":
                            result.source_level,
                        "rating_available":
                            result.rating_available,
                        "temporal_validity_pass":
                            temporal_pass,
                        "overall_validation_pass":
                            overall_pass,
                    }
                )

            except Exception as error:
                record["error_type"] = (
                    type(error).__name__
                )

                record["error_message"] = str(
                    error
                )

            records.append(record)

    return pd.DataFrame(records)


def validate_pair_integrity(
    audit: pd.DataFrame,
    match_count: int,
) -> None:
    expected_row_count = (
        match_count * 2
    )

    if len(audit) != expected_row_count:
        raise AssertionError(
            "Provider audit row count differs from "
            f"the expected value: {len(audit)} vs "
            f"{expected_row_count}."
        )

    role_counts = (
        audit.groupby("event_id")[
            "venue_role"
        ]
        .nunique()
    )

    if not role_counts.eq(2).all():
        raise AssertionError(
            "One or more matches do not contain both "
            "home and away provider results."
        )

    request_counts = (
        audit.groupby("event_id")
        .size()
    )

    if not request_counts.eq(2).all():
        raise AssertionError(
            "One or more matches do not contain exactly "
            "two provider requests."
        )


def build_match_prior_table(
    matches: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    successful = audit[
        audit["overall_validation_pass"]
    ].copy()

    home = (
        successful[
            successful["venue_role"].eq(
                "home"
            )
        ]
        .set_index("event_id")
        .sort_index()
    )

    away = (
        successful[
            successful["venue_role"].eq(
                "away"
            )
        ]
        .set_index("event_id")
        .sort_index()
    )

    if not home.index.equals(
        away.index
    ):
        raise AssertionError(
            "Home and away rating-prior event "
            "populations do not match."
        )

    expected_event_ids = set(
        matches["event_id"]
    )

    if set(home.index) != expected_event_ids:
        raise AssertionError(
            "Successful provider coverage does not "
            "match the complete observation population."
        )

    output = (
        matches
        .set_index("event_id")
        .sort_index()
        [
            [
                "date",
                "home_team",
                "away_team",
            ]
        ]
        .copy()
    )

    output[
        "home_rating_prior"
    ] = home["rating"]

    output[
        "away_rating_prior"
    ] = away["rating"]

    output[
        "rating_prior_diff"
    ] = (
        output["home_rating_prior"]
        - output["away_rating_prior"]
    )

    output[
        "rating_prior_source"
    ] = "clubelo"

    output[
        "rating_prior_available"
    ] = True

    output[
        "home_rating_effective_from"
    ] = home["effective_from"]

    output[
        "home_rating_effective_to"
    ] = home["effective_to"]

    output[
        "away_rating_effective_from"
    ] = away["effective_from"]

    output[
        "away_rating_effective_to"
    ] = away["effective_to"]

    return output.reset_index()


def main() -> None:
    matches = load_match_requests()

    repository = ClubEloRepository(
        cache_directory=CACHE_DIRECTORY
    )

    provider = ClubEloRatingPriorProvider(
        repository=repository
    )

    audit = build_provider_audit(
        matches=matches,
        provider=provider,
    )

    validate_pair_integrity(
        audit=audit,
        match_count=len(matches),
    )

    failures = audit[
        ~audit[
            "overall_validation_pass"
        ]
    ].copy()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_DIRECTORY
        / "rating_prior_request_audit.csv",
        index=False,
    )

    failures.to_csv(
        OUTPUT_DIRECTORY
        / "rating_prior_failures.csv",
        index=False,
    )

    if not failures.empty:
        print(
            failures[
                [
                    "event_id",
                    "match_date",
                    "venue_role",
                    "requested_team_name",
                    "error_type",
                    "error_message",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

        raise AssertionError(
            "Rating-prior provider did not achieve "
            "complete observation coverage."
        )

    match_prior_table = (
        build_match_prior_table(
            matches=matches,
            audit=audit,
        )
    )

    match_prior_table.to_csv(
        OUTPUT_DIRECTORY
        / "match_rating_priors.csv",
        index=False,
    )

    if not np.allclose(
        match_prior_table[
            "rating_prior_diff"
        ].to_numpy(dtype=float),
        (
            match_prior_table[
                "home_rating_prior"
            ].to_numpy(dtype=float)
            - match_prior_table[
                "away_rating_prior"
            ].to_numpy(dtype=float)
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Rating-prior differences are not equal "
            "to home minus away ratings."
        )

    if not match_prior_table[
        "rating_prior_available"
    ].all():
        raise AssertionError(
            "One or more match priors are unavailable."
        )

    if not match_prior_table[
        "rating_prior_source"
    ].eq("clubelo").all():
        raise AssertionError(
            "Unexpected rating-prior source."
        )

    print("Rating Prior Provider Validation")
    print("=" * 76)
    print()
    print(
        f"Observation matches: "
        f"{len(matches)}"
    )
    print(
        "Provider requests: "
        f"{len(audit)}"
    )
    print(
        "Successful requests: "
        f"{int(audit['overall_validation_pass'].sum())}"
    )
    print(
        "Failed requests: "
        f"{len(failures)}"
    )
    print()

    print("Rating Prior Summary")
    print("-" * 76)
    print(
        match_prior_table[
            [
                "home_rating_prior",
                "away_rating_prior",
                "rating_prior_diff",
            ]
        ]
        .describe()
        .to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )

    print()
    print("Provider contract: PASS")
    print("Identity integration: PASS")
    print("Repository integration: PASS")
    print("Complete match coverage: PASS")
    print("Temporal validity: PASS")
    print("Rating-difference derivation: PASS")
    print("Provenance propagation: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()