#build_sofascore_league_fixture_snapshot

# build_sofascore_league_fixture_snapshot.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.sofascore_utils import get_json


SOFASCORE_BASE_URL = "https://www.sofascore.com/api/v1"


OUTPUT_COLUMNS = [
    "fixture_id",
    "competition",
    "season",
    "matchday",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "source_url",
]


@dataclass(frozen=True)
class SofascoreLeagueFixtureSnapshotConfig:
    competition: str
    season: str

    unique_tournament_id: int
    season_id: int

    participant_count: int
    matchday_count: int
    fixture_count: int

    timezone_name: str

    output_path: Path

    @property
    def matches_per_matchday(self) -> int:
        return self.participant_count // 2

    @property
    def matches_per_team(self) -> int:
        return 2 * (self.participant_count - 1)

    def validate(self) -> None:
        if self.participant_count < 2:
            raise ValueError(
                "participant_count must be at least 2."
            )

        if self.participant_count % 2 != 0:
            raise ValueError(
                "participant_count must be even."
            )

        expected_matchdays = (
            2 * (self.participant_count - 1)
        )

        if self.matchday_count != expected_matchdays:
            raise ValueError(
                "matchday_count is inconsistent with a "
                "double round robin: "
                f"expected {expected_matchdays}, "
                f"received {self.matchday_count}."
            )

        expected_fixtures = (
            self.participant_count
            * (self.participant_count - 1)
        )

        if self.fixture_count != expected_fixtures:
            raise ValueError(
                "fixture_count is inconsistent with a "
                "double round robin: "
                f"expected {expected_fixtures}, "
                f"received {self.fixture_count}."
            )

        try:
            ZoneInfo(self.timezone_name)
        except Exception as exc:
            raise ValueError(
                f"Invalid timezone_name: "
                f"{self.timezone_name!r}."
            ) from exc

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical Sofascore fixture snapshot "
            "for a domestic league season."
        )
    )

    parser.add_argument(
        "--competition",
        required=True,
    )

    parser.add_argument(
        "--season",
        required=True,
    )

    parser.add_argument(
        "--unique-tournament-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--season-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--participant-count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--matchday-count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--fixture-count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--timezone-name",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    return parser.parse_args()

def fetch_round(
    config: SofascoreLeagueFixtureSnapshotConfig,
    matchday: int,
) -> list[dict[str, Any]]:
    if not 1 <= matchday <= config.matchday_count:
        raise ValueError(
            f"Invalid matchday {matchday}; expected "
            f"1..{config.matchday_count}."
        )

    url = (
        f"{SOFASCORE_BASE_URL}/"
        f"unique-tournament/{config.unique_tournament_id}/"
        f"season/{config.season_id}/"
        f"events/round/{matchday}"
    )

    data = get_json(url)

    events = data.get("events")

    if not isinstance(events, list):
        raise ValueError(
            f"Sofascore round {matchday} response does not "
            "contain an events list."
        )

    events = [
        event
        for event in events
        if isinstance(event, dict)
    ]

    if len(events) != config.matches_per_matchday:
        raise ValueError(
            f"Matchday {matchday} contains {len(events)} "
            f"fixtures; expected "
            f"{config.matches_per_matchday}."
        )

    return events

def canonicalize_event(
    config: SofascoreLeagueFixtureSnapshotConfig,
    matchday: int,
    event: dict[str, Any],
) -> dict[str, object]:
    event_id = event.get("id")

    if event_id is None:
        raise ValueError(
            f"Matchday {matchday} contains an event "
            "without a Sofascore event ID."
        )

    round_info = event.get("roundInfo", {})
    source_matchday = round_info.get("round")

    if source_matchday is not None:
        try:
            source_matchday = int(source_matchday)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Event {event_id} has invalid round: "
                f"{source_matchday!r}."
            ) from exc

        if source_matchday != matchday:
            raise ValueError(
                f"Event {event_id} belongs to round "
                f"{source_matchday}, expected {matchday}."
            )

    home_team = str(
        event.get("homeTeam", {}).get("name", "")
    ).strip()

    away_team = str(
        event.get("awayTeam", {}).get("name", "")
    ).strip()

    if not home_team or not away_team:
        raise ValueError(
            f"Event {event_id} is missing a team name."
        )

    if home_team == away_team:
        raise ValueError(
            f"Event {event_id} has the same home and "
            "away team."
        )

    start_timestamp = event.get("startTimestamp")

    if start_timestamp is None:
        raise ValueError(
            f"Event {event_id} is missing startTimestamp."
        )

    try:
        kickoff = datetime.fromtimestamp(
            int(start_timestamp),
            tz=ZoneInfo(
                config.timezone_name
            ),
        )
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError(
            f"Event {event_id} has invalid "
            f"startTimestamp: {start_timestamp!r}."
        ) from exc

    source_url = (
        f"{SOFASCORE_BASE_URL}/"
        f"unique-tournament/{config.unique_tournament_id}/"
        f"season/{config.season_id}/"
        f"events/round/{matchday}"
    )

    return {
        "fixture_id": f"sofascore_{event_id}",
        "competition": config.competition,
        "season": config.season,
        "matchday": matchday,
        "match_date": kickoff.date().isoformat(),
        "kickoff_time": kickoff.strftime("%H:%M"),
        "home_team": home_team,
        "away_team": away_team,
        "source_url": source_url,
    }

def build_fixture_snapshot(
    config: SofascoreLeagueFixtureSnapshotConfig,
) -> pd.DataFrame:
    config.validate()

    rows: list[dict[str, object]] = []

    for matchday in range(
        1,
        config.matchday_count + 1,
    ):
        events = fetch_round(
            config,
            matchday,
        )

        rows.extend(
            canonicalize_event(
                config,
                matchday,
                event,
            )
            for event in events
        )

    return pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )


def validate_fixture_snapshot(
    config: SofascoreLeagueFixtureSnapshotConfig,
    fixtures: pd.DataFrame,
) -> None:
    if len(fixtures) != config.fixture_count:
        raise ValueError(
            f"Fixture snapshot contains {len(fixtures)} rows; "
            f"expected {config.fixture_count}."
        )

    if fixtures["fixture_id"].duplicated().any():
        raise ValueError(
            "Fixture snapshot contains duplicate fixture IDs."
        )

    teams = set(fixtures["home_team"]) | set(
        fixtures["away_team"]
    )

    if len(teams) != config.participant_count:
        raise ValueError(
            f"Fixture snapshot contains {len(teams)} teams; "
            f"expected {config.participant_count}."
        )

    matchday_counts = (
        fixtures.groupby("matchday")
        .size()
    )

    expected_matchdays = set(
        range(
            1,
            config.matchday_count + 1,
        )
    )

    observed_matchdays = set(
        matchday_counts.index
    )

    if observed_matchdays != expected_matchdays:
        raise ValueError(
            "Fixture snapshot does not contain exactly "
            "the expected matchdays."
        )

    invalid_matchdays = (
        matchday_counts[
            matchday_counts
            != config.matches_per_matchday
        ]
    )

    if not invalid_matchdays.empty:
        raise ValueError(
            "One or more matchdays contain an invalid "
            "fixture count: "
            f"{invalid_matchdays.to_dict()}."
        )

    directed_pairs = list(
        zip(
            fixtures["home_team"],
            fixtures["away_team"],
        )
    )

    if len(set(directed_pairs)) != config.fixture_count:
        raise ValueError(
            "Fixture snapshot does not contain exactly "
            "one occurrence of every directed team pairing."
        )

    for team in sorted(teams):
        home_count = int(
            (fixtures["home_team"] == team).sum()
        )

        away_count = int(
            (fixtures["away_team"] == team).sum()
        )

        expected_home = (
            config.participant_count - 1
        )

        expected_away = (
            config.participant_count - 1
        )

        if (
            home_count != expected_home
            or away_count != expected_away
        ):
            raise ValueError(
                f"{team!r} has invalid home/away counts: "
                f"home={home_count}, away={away_count}; "
                f"expected {expected_home}/{expected_away}."
            )

        total_count = home_count + away_count

        if total_count != config.matches_per_team:
            raise ValueError(
                f"{team!r} has {total_count} fixtures; "
                f"expected {config.matches_per_team}."
            )

def write_fixture_snapshot(
    config: SofascoreLeagueFixtureSnapshotConfig,
    fixtures: pd.DataFrame,
) -> None:
    validate_fixture_snapshot(
        config,
        fixtures,
    )

    config.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixtures.to_csv(
        config.output_path,
        index=False,
    )

def main() -> None:
    arguments = parse_arguments()

    config = SofascoreLeagueFixtureSnapshotConfig(
        competition=arguments.competition,
        season=arguments.season,
        unique_tournament_id=(
            arguments.unique_tournament_id
        ),
        season_id=arguments.season_id,
        participant_count=(
            arguments.participant_count
        ),
        matchday_count=(
            arguments.matchday_count
        ),
        fixture_count=arguments.fixture_count,
        timezone_name=arguments.timezone_name,
        output_path=arguments.output_path,
    )

    fixtures = build_fixture_snapshot(
        config
    )

    write_fixture_snapshot(
        config=config,
        fixtures=fixtures,
    )

    print()
    print(
        f"{config.competition.upper()} "
        f"{config.season} FIXTURE SNAPSHOT"
    )
    print("=" * 72)

    print("Fixtures:", len(fixtures))
    print(
        "Teams:",
        len(
            set(fixtures["home_team"])
            | set(fixtures["away_team"])
        ),
    )
    print(
        "Matchdays:",
        fixtures["matchday"].nunique(),
    )
    print(
        "Output:",
        config.output_path,
    )

    print()
    print("Fixture snapshot: PASS")


if __name__ == "__main__":
    main()