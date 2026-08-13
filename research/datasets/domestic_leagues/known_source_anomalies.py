#known_source_anomalies

from __future__ import annotations

from dataclasses import dataclass


Fixture = tuple[str, str]
AnomalyKey = tuple[str, int]


@dataclass(frozen=True)
class KnownSourceAnomaly:
    """
    A verified discrepancy between the official expected league
    schedule and the completed regular-season fixtures available
    in the raw source dataset.
    """

    competition_key: str
    season_start_year: int
    missing_fixtures: tuple[Fixture, ...]
    source_event_ids: tuple[int, ...]
    description: str

    @property
    def expected_missing_match_count(self) -> int:
        return len(self.missing_fixtures)


KNOWN_SOURCE_ANOMALIES: dict[
    AnomalyKey,
    KnownSourceAnomaly,
] = {
    (
        "bundesliga",
        2021,
    ): KnownSourceAnomaly(
        competition_key="bundesliga",
        season_start_year=2021,
        missing_fixtures=(
            (
                "VfL Bochum 1848",
                "Borussia M'gladbach",
            ),
        ),
        source_event_ids=(
            9594319,
        ),
        description=(
            "The raw source dataset contains the directed fixture "
            "VfL Bochum 1848 vs Borussia M'gladbach only as an "
            "abandoned event. Excluding non-final events leaves "
            "one expected regular-season fixture absent."
        ),
    ),
    (
        "ligue_1",
        2025,
    ): KnownSourceAnomaly(
        competition_key="ligue_1",
        season_start_year=2025,
        missing_fixtures=(
            (
                "Nantes",
                "Toulouse",
            ),
        ),
        source_event_ids=(
            14061912,
        ),
        description=(
            "The raw source dataset contains the directed fixture "
            "Nantes vs Toulouse only as an abandoned event. "
            "Excluding non-final events leaves one expected "
            "regular-season fixture absent."
        ),
    ),
}


def get_known_source_anomaly(
    competition_key: str,
    season_start_year: int,
) -> KnownSourceAnomaly | None:
    """
    Return the registered source anomaly for a league-season,
    when one exists.
    """

    return KNOWN_SOURCE_ANOMALIES.get(
        (
            competition_key,
            season_start_year,
        )
    )

def validate_known_source_anomaly(
    anomaly: KnownSourceAnomaly,
    missing_fixtures: tuple[Fixture, ...],
    unexpected_fixtures: tuple[Fixture, ...],
    duplicate_fixtures: tuple[
        tuple[str, str, int],
        ...,
    ],
) -> None:
    """
    Confirm that a failed fixture-integrity report exactly matches
    a registered source anomaly.

    No additional missing, unexpected, or duplicate fixtures are
    accepted.
    """

    observed_missing = tuple(
        sorted(missing_fixtures)
    )

    expected_missing = tuple(
        sorted(anomaly.missing_fixtures)
    )

    if observed_missing != expected_missing:
        raise ValueError(
            "Observed missing fixtures do not exactly match the "
            "registered source anomaly. "
            f"Expected: {list(expected_missing)}. "
            f"Observed: {list(observed_missing)}."
        )

    if unexpected_fixtures:
        raise ValueError(
            "A registered source anomaly cannot excuse unexpected "
            "fixtures. Observed unexpected fixtures: "
            f"{list(unexpected_fixtures)}"
        )

    if duplicate_fixtures:
        raise ValueError(
            "A registered source anomaly cannot excuse duplicate "
            "fixtures. Observed duplicate fixtures: "
            f"{list(duplicate_fixtures)}"
        )