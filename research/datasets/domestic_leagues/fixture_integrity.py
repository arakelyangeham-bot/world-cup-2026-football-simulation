#fixture_integrity

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations

import pandas as pd


Fixture = tuple[str, str]


@dataclass(frozen=True)
class FixtureIntegrityReport:
    """
    Fixture-level comparison between the expected and observed
    schedules of a double round-robin domestic league.
    """

    team_names: tuple[str, ...]
    expected_fixture_count: int
    observed_match_count: int
    observed_unique_fixture_count: int
    missing_fixtures: tuple[Fixture, ...]
    unexpected_fixtures: tuple[Fixture, ...]
    duplicate_fixtures: tuple[
        tuple[str, str, int],
        ...,
    ]

    @property
    def team_count(self) -> int:
        return len(self.team_names)

    @property
    def missing_fixture_count(self) -> int:
        return len(self.missing_fixtures)

    @property
    def unexpected_fixture_count(self) -> int:
        return len(self.unexpected_fixtures)

    @property
    def duplicate_fixture_count(self) -> int:
        return len(self.duplicate_fixtures)

    @property
    def passed(self) -> bool:
        return (
            self.observed_match_count
            == self.expected_fixture_count
            and self.missing_fixture_count == 0
            and self.unexpected_fixture_count == 0
            and self.duplicate_fixture_count == 0
        )


def extract_team_names(
    matches: pd.DataFrame,
) -> tuple[str, ...]:
    """
    Extract the complete set of clubs appearing in the supplied
    match data.
    """

    required_columns = {
        "home_team",
        "away_team",
    }

    missing_columns = (
        required_columns - set(matches.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fixture-integrity analysis requires columns: "
            f"{sorted(required_columns)}. Missing: "
            f"{sorted(missing_columns)}"
        )

    home_teams = (
        matches["home_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    away_teams = (
        matches["away_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    teams = sorted(
        set(home_teams)
        | set(away_teams)
    )

    if not teams:
        raise ValueError(
            "No clubs were available for fixture-integrity "
            "analysis."
        )

    return tuple(teams)


def build_expected_fixtures(
    team_names: tuple[str, ...],
) -> tuple[Fixture, ...]:
    """
    Build every directed fixture expected in a double round-robin
    competition.

    For each pair of clubs, both home-away orientations are
    expected exactly once.
    """

    if len(team_names) < 2:
        raise ValueError(
            "At least two clubs are required to build an "
            "expected fixture schedule."
        )

    if len(set(team_names)) != len(team_names):
        raise ValueError(
            "Team names must be unique when building the "
            "expected fixture schedule."
        )

    return tuple(
        sorted(
            permutations(
                team_names,
                2,
            )
        )
    )


def build_observed_fixture_counts(
    matches: pd.DataFrame,
) -> Counter[Fixture]:
    """
    Count observed directed fixtures.
    """

    fixture_counts: Counter[Fixture] = Counter()

    for row in matches.itertuples(
        index=False
    ):
        home_team = str(
            row.home_team
        ).strip()

        away_team = str(
            row.away_team
        ).strip()

        if not home_team or not away_team:
            raise ValueError(
                "Observed fixtures contain an empty team name."
            )

        if home_team == away_team:
            raise ValueError(
                "Observed fixture contains the same home and "
                f"away club: {home_team!r}."
            )

        fixture_counts[
            (
                home_team,
                away_team,
            )
        ] += 1

    return fixture_counts


def analyze_fixture_integrity(
    completed_matches: pd.DataFrame,
    team_names: tuple[str, ...] | None = None,
) -> FixtureIntegrityReport:
    """
    Compare completed matches with the expected directed schedule
    of a double round-robin league.
    """

    if completed_matches.empty:
        raise ValueError(
            "Cannot analyze fixture integrity for an empty "
            "completed-match dataframe."
        )

    if team_names is None:
        team_names = extract_team_names(
            completed_matches
        )

    expected_fixtures = set(
        build_expected_fixtures(
            team_names
        )
    )

    observed_counts = (
        build_observed_fixture_counts(
            completed_matches
        )
    )

    observed_fixtures = set(
        observed_counts
    )

    missing_fixtures = tuple(
        sorted(
            expected_fixtures
            - observed_fixtures
        )
    )

    unexpected_fixtures = tuple(
        sorted(
            observed_fixtures
            - expected_fixtures
        )
    )

    duplicate_fixtures = tuple(
        sorted(
            (
                home_team,
                away_team,
                count,
            )
            for (
                home_team,
                away_team,
            ), count in observed_counts.items()
            if count > 1
        )
    )

    return FixtureIntegrityReport(
        team_names=tuple(
            sorted(team_names)
        ),
        expected_fixture_count=len(
            expected_fixtures
        ),
        observed_match_count=len(
            completed_matches
        ),
        observed_unique_fixture_count=len(
            observed_fixtures
        ),
        missing_fixtures=missing_fixtures,
        unexpected_fixtures=(
            unexpected_fixtures
        ),
        duplicate_fixtures=duplicate_fixtures,
    )


def print_fixture_integrity_report(
    report: FixtureIntegrityReport,
) -> None:
    """
    Print a human-readable fixture-integrity report.
    """

    print("Fixture Integrity Report")
    print("========================")
    print(
        f"Clubs detected: "
        f"{report.team_count}"
    )
    print(
        f"Expected directed fixtures: "
        f"{report.expected_fixture_count}"
    )
    print(
        f"Observed match rows: "
        f"{report.observed_match_count}"
    )
    print(
        f"Observed unique fixtures: "
        f"{report.observed_unique_fixture_count}"
    )
    print()

    print(
        f"Missing fixtures: "
        f"{report.missing_fixture_count}"
    )

    for home_team, away_team in (
        report.missing_fixtures
    ):
        print(
            f"  {home_team} vs {away_team}"
        )

    print()

    print(
        f"Unexpected fixtures: "
        f"{report.unexpected_fixture_count}"
    )

    for home_team, away_team in (
        report.unexpected_fixtures
    ):
        print(
            f"  {home_team} vs {away_team}"
        )

    print()

    print(
        f"Duplicate fixtures: "
        f"{report.duplicate_fixture_count}"
    )

    for (
        home_team,
        away_team,
        count,
    ) in report.duplicate_fixtures:
        print(
            f"  {home_team} vs {away_team}: "
            f"{count} rows"
        )

    print()

    print("Fixture Integrity Result")
    print("------------------------")

    if report.passed:
        print("PASSED")
    else:
        print("FAILED")

def print_missing_fixture_source_events(
    raw: pd.DataFrame,
    missing_fixtures: tuple[
        tuple[str, str],
        ...,
    ],
) -> None:
    """
    Print every raw source event involving each missing directed
    fixture.

    Both home-away orientations are included so postponed,
    abandoned, replayed, or replacement events can be inspected
    together.
    """

    if not missing_fixtures:
        return

    source_columns = [
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status_code",
        "status_desc",
        "stage",
        "round",
        "round_number",
    ]

    available_columns = [
        column
        for column in source_columns
        if column in raw.columns
    ]

    print("Missing Fixture Source Traces")
    print("=============================")

    for home_team, away_team in missing_fixtures:
        fixture_mask = (
            (
                raw["home_team"]
                .astype(str)
                .str.strip()
                .eq(home_team)
                &
                raw["away_team"]
                .astype(str)
                .str.strip()
                .eq(away_team)
            )
            |
            (
                raw["home_team"]
                .astype(str)
                .str.strip()
                .eq(away_team)
                &
                raw["away_team"]
                .astype(str)
                .str.strip()
                .eq(home_team)
            )
        )

        candidates = (
            raw.loc[
                fixture_mask,
                available_columns,
            ]
            .copy()
            .sort_values(
                ["date", "event_id"],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        print()
        print(
            f"Missing directed fixture: "
            f"{home_team} vs {away_team}"
        )

        if candidates.empty:
            print(
                "No raw source events were found for "
                "this club pairing."
            )
            continue

        print(
            candidates.to_string(
                index=False
            )
        )

    print()