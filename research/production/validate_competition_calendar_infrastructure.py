#validate_competition_calendar_infrastructure

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fixture_generation import (
    RoundRobinFixtureGenerator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_072_competition_calendar_infrastructure"
)

FIXTURE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "calendar_fixture_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


TEAM_COUNT = 20
START_DATE = date(2025, 8, 16)
DAYS_BETWEEN_MATCHDAYS = 7


def build_participants() -> list[str]:
    return [
        f"Club {index:02d}"
        for index in range(
            1,
            TEAM_COUNT + 1,
        )
    ]


def build_calendar_fixtures():
    generator = (
        RoundRobinFixtureGenerator()
    )

    return generator.generate(
        participants=build_participants(),
        double_round_robin=True,
        competition_id=(
            "calendar_validation"
        ),
        start_date=START_DATE,
        days_between_matchdays=(
            DAYS_BETWEEN_MATCHDAYS
        ),
    )


def validate_calendar_fixtures(
    fixtures,
) -> pd.DataFrame:
    if len(fixtures) != 380:
        raise AssertionError(
            "Expected 380 double-round-robin "
            f"fixtures, found {len(fixtures)}."
        )

    if any(
        fixture.match_date is None
        for fixture in fixtures
    ):
        raise AssertionError(
            "One or more calendar-aware fixtures "
            "have no date."
        )

    if not all(
        fixture.is_calendar_aware
        for fixture in fixtures
    ):
        raise AssertionError(
            "Fixture calendar-awareness property "
            "failed."
        )

    records = [
        {
            "fixture_id":
                fixture.fixture_id,
            "matchday":
                fixture.matchday,
            "match_date":
                fixture.match_date,
            "home_team":
                fixture.home_team,
            "away_team":
                fixture.away_team,
            "leg":
                fixture.leg,
        }
        for fixture in fixtures
    ]

    audit = pd.DataFrame(records)

    if audit["fixture_id"].duplicated().any():
        raise AssertionError(
            "Generated fixture IDs are not unique."
        )

    if audit["matchday"].min() != 1:
        raise AssertionError(
            "First matchday is not 1."
        )

    if audit["matchday"].max() != 38:
        raise AssertionError(
            "Final matchday is not 38."
        )

    matches_per_matchday = (
        audit.groupby(
            "matchday"
        ).size()
    )

    if not matches_per_matchday.eq(10).all():
        raise AssertionError(
            "Every 20-team matchday should contain "
            "10 fixtures."
        )

    dates_per_matchday = (
        audit.groupby(
            "matchday"
        )["match_date"]
        .nunique()
    )

    if not dates_per_matchday.eq(1).all():
        raise AssertionError(
            "Fixtures on one matchday do not share "
            "one date."
        )

    matchday_dates = (
        audit[
            [
                "matchday",
                "match_date",
            ]
        ]
        .drop_duplicates()
        .sort_values("matchday")
        .reset_index(drop=True)
    )

    for row in matchday_dates.itertuples(
        index=False
    ):
        expected_date = (
            START_DATE
            + timedelta(
                days=(
                    (row.matchday - 1)
                    * DAYS_BETWEEN_MATCHDAYS
                )
            )
        )

        if row.match_date != expected_date:
            raise AssertionError(
                "Unexpected date for matchday "
                f"{row.matchday}: "
                f"{row.match_date} vs "
                f"{expected_date}."
            )

    if not audit.loc[
        audit["matchday"].le(19),
        "leg",
    ].eq(1).all():
        raise AssertionError(
            "First-leg matchdays contain an "
            "unexpected leg value."
        )

    if not audit.loc[
        audit["matchday"].ge(20),
        "leg",
    ].eq(2).all():
        raise AssertionError(
            "Second-leg matchdays contain an "
            "unexpected leg value."
        )

    pair_counts = (
        audit.assign(
            pairing=audit.apply(
                lambda row: tuple(
                    sorted(
                        [
                            row["home_team"],
                            row["away_team"],
                        ]
                    )
                ),
                axis=1,
            )
        )
        .groupby("pairing")
        .size()
    )

    if not pair_counts.eq(2).all():
        raise AssertionError(
            "Every team pairing should occur "
            "exactly twice."
        )

    return audit


def validate_legacy_schedule() -> None:
    generator = (
        RoundRobinFixtureGenerator()
    )

    fixtures = generator.generate(
        participants=[
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
        ],
        double_round_robin=False,
        competition_id=(
            "legacy_validation"
        ),
    )

    if any(
        fixture.match_date is not None
        for fixture in fixtures
    ):
        raise AssertionError(
            "Legacy schedule unexpectedly received "
            "calendar dates."
        )

    if any(
        fixture.is_calendar_aware
        for fixture in fixtures
    ):
        raise AssertionError(
            "Legacy fixture incorrectly reports "
            "calendar awareness."
        )


def build_metadata(
    audit: pd.DataFrame,
) -> dict[str, object]:
    unique_dates = (
        audit["match_date"]
        .drop_duplicates()
        .sort_values()
    )

    return {
        "study_id": "072",
        "study_name": (
            "Competition Calendar Infrastructure"
        ),
        "team_count": TEAM_COUNT,
        "fixture_count": int(
            len(audit)
        ),
        "matchday_count": int(
            audit["matchday"].nunique()
        ),
        "matches_per_matchday": 10,
        "start_date": (
            unique_dates.iloc[0]
            .isoformat()
        ),
        "end_date": (
            unique_dates.iloc[-1]
            .isoformat()
        ),
        "days_between_matchdays": (
            DAYS_BETWEEN_MATCHDAYS
        ),
        "calendar_dates_complete_pass": True,
        "matchday_date_consistency_pass": True,
        "chronological_progression_pass": True,
        "double_round_robin_pass": True,
        "legacy_compatibility_pass": True,
        "overall_result": "PASS",
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 072 — Competition Calendar Infrastructure

## Purpose

Introduce calendar-aware scheduled fixtures while preserving
legacy fixture-generation behavior.

## Validation competition

- Teams: {metadata["team_count"]}
- Fixtures: {metadata["fixture_count"]}
- Matchdays: {metadata["matchday_count"]}
- Matches per matchday:
  {metadata["matches_per_matchday"]}
- Start date: {metadata["start_date"]}
- End date: {metadata["end_date"]}
- Days between matchdays:
  {metadata["days_between_matchdays"]}

## Calendar policy

All fixtures on one matchday share the same calendar date.

Successive matchdays are separated by a configurable number
of days. This study uses seven days.

This is scheduling infrastructure rather than a recreation of
a specific historical Premier League calendar.

## Validation

- Complete calendar-date coverage: PASS
- One date per matchday: PASS
- Chronological date progression: PASS
- Configurable matchday spacing: PASS
- 380-fixture double round robin: PASS
- Ten fixtures per matchday: PASS
- First- and second-leg assignment: PASS
- Two matches per team pairing: PASS
- Unique fixture identifiers: PASS
- Legacy no-date generation: PASS

## Boundary established

`ScheduledFixture` can now carry a real calendar date through
the competition framework.

Passing that date into the production predictor remains the
responsibility of the next integration study.

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    fixtures = build_calendar_fixtures()

    audit = validate_calendar_fixtures(
        fixtures
    )

    validate_legacy_schedule()

    metadata = build_metadata(
        audit
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_audit = audit.copy()

    output_audit["match_date"] = (
        pd.to_datetime(
            output_audit["match_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    output_audit.to_csv(
        FIXTURE_AUDIT_PATH,
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

    print(
        "Study 072 — Competition Calendar "
        "Infrastructure"
    )
    print("=" * 76)
    print()
    print(
        f"Teams: {TEAM_COUNT}"
    )
    print(
        f"Fixtures: {len(audit)}"
    )
    print(
        "Matchdays: "
        f"{audit['matchday'].nunique()}"
    )
    print(
        f"Start date: "
        f"{metadata['start_date']}"
    )
    print(
        f"End date: "
        f"{metadata['end_date']}"
    )
    print(
        "Days between matchdays: "
        f"{DAYS_BETWEEN_MATCHDAYS}"
    )
    print()
    print("Calendar-date coverage: PASS")
    print("One date per matchday: PASS")
    print("Chronological progression: PASS")
    print("Double round robin: PASS")
    print("Fixture pairing contract: PASS")
    print("Unique fixture IDs: PASS")
    print("Legacy compatibility: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()