#build_premier_league_2026_27_fixture_snapshot

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


SOURCE_URL = (
    "https://www.premierleague.com/en/news/"
    "4675097/all-380-fixtures-for-202627-premier-league-season"
)

OUTPUT_DIRECTORY = Path(
    "outputs/premier_league_2026_27_bootstrap"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "premier_league_2026_27_fixtures.csv"
)

RAW_SNAPSHOT_PATH = (
    OUTPUT_DIRECTORY
    / "premier_league_2026_27_fixture_source_snapshot.txt"
)

COMPETITION = "Premier League"
SEASON = "2026-27"


# ------------------------------------------------------------------
# Premier League published names -> our canonical repository names.
# ------------------------------------------------------------------

TEAM_NAME_OVERRIDES = {
    "AFC Bournemouth": "Bournemouth",
    "Liverpool": "Liverpool FC",
}


EXPECTED_TEAMS = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds United",
    "Liverpool FC",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
}


DATE_PATTERN = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday)\s+"
    r"(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)"
    r"(?:\s+(\d{4}))?$"
)


FIXTURE_PATTERN = re.compile(
    r"^(?:(\d{2}:\d{2})\s+)?"
    r"(.+?)\s+v\s+(.+?)"
    r"(?:\s+\([^)]*\))?"
    r"(?:\*+)?$"
)


def normalize_team_name(name: str) -> str:
    name = name.strip()

    return TEAM_NAME_OVERRIDES.get(
        name,
        name,
    )


def clean_text(text: str) -> str:
    return " ".join(
        text.replace("\xa0", " ").split()
    )


def fetch_source_lines() -> list[str]:
    response = requests.get(
        SOURCE_URL,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; football-research-fixture-snapshot/1.0)"
            )
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    lines = []

    for element in soup.stripped_strings:
        line = clean_text(element)

        if line:
            lines.append(line)

    return lines


def parse_date_heading(
    line: str,
    current_year: int,
) -> tuple[datetime.date | None, int]:
    match = DATE_PATTERN.match(line)

    if match is None:
        return None, current_year

    _, day, month, explicit_year = (
        match.groups()
    )

    if explicit_year is not None:
        current_year = int(explicit_year)

    parsed = datetime.strptime(
        f"{day} {month} {current_year}",
        "%d %B %Y",
    ).date()

    return parsed, current_year


def parse_fixtures(
    lines: list[str],
) -> pd.DataFrame:
    fixtures = []

    current_date = None
    current_year = 2026

    for line in lines:
        parsed_date, current_year = (
            parse_date_heading(
                line,
                current_year,
            )
        )

        if parsed_date is not None:
            current_date = parsed_date
            continue

        if current_date is None:
            continue

        match = FIXTURE_PATTERN.match(line)

        if match is None:
            continue

        kickoff_time, home_team, away_team = (
            match.groups()
        )

        home_team = normalize_team_name(
            home_team
        )

        away_team = normalize_team_name(
            away_team
        )

        # This prevents prose containing " v " from being
        # accidentally interpreted as a fixture.
        if (
            home_team not in EXPECTED_TEAMS
            or away_team not in EXPECTED_TEAMS
        ):
            continue

        fixture_number = len(fixtures) + 1

        # The official release is ordered by league round.
        # 20 clubs = 10 fixtures per matchday.
        matchday = (
            (fixture_number - 1) // 10
        ) + 1

        fixtures.append(
            {
                "fixture_id": (
                    f"pl_2026_27_"
                    f"{fixture_number:03d}"
                ),
                "competition": COMPETITION,
                "season": SEASON,
                "matchday": matchday,
                "match_date": (
                    current_date.isoformat()
                ),
                "kickoff_time": (
                    kickoff_time or ""
                ),
                "home_team": home_team,
                "away_team": away_team,
                "source_url": SOURCE_URL,
            }
        )

    return pd.DataFrame(fixtures)


def validate_fixture_frame(
    fixtures: pd.DataFrame,
) -> None:
    if len(fixtures) != 380:
        raise ValueError(
            "Expected 380 fixtures, "
            f"found {len(fixtures)}."
        )

    teams = set(
        fixtures["home_team"]
    ) | set(
        fixtures["away_team"]
    )

    if teams != EXPECTED_TEAMS:
        missing = sorted(
            EXPECTED_TEAMS - teams
        )

        unexpected = sorted(
            teams - EXPECTED_TEAMS
        )

        raise ValueError(
            "Participant mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    if fixtures["fixture_id"].duplicated().any():
        raise ValueError(
            "Duplicate fixture IDs found."
        )

    if fixtures["matchday"].min() != 1:
        raise ValueError(
            "First matchday is not 1."
        )

    if fixtures["matchday"].max() != 38:
        raise ValueError(
            "Final matchday is not 38."
        )

    matches_per_matchday = (
        fixtures
        .groupby("matchday")
        .size()
    )

    if not matches_per_matchday.eq(10).all():
        raise ValueError(
            "Every matchday must contain "
            "exactly 10 fixtures."
        )

    appearances = pd.concat(
        [
            fixtures[
                ["matchday", "home_team"]
            ].rename(
                columns={
                    "home_team": "team"
                }
            ),
            fixtures[
                ["matchday", "away_team"]
            ].rename(
                columns={
                    "away_team": "team"
                }
            ),
        ],
        ignore_index=True,
    )

    duplicate_matchday_team = (
        appearances
        .duplicated(
            subset=[
                "matchday",
                "team",
            ]
        )
        .any()
    )

    if duplicate_matchday_team:
        raise ValueError(
            "At least one club appears more "
            "than once on one matchday."
        )

    matches_per_team = (
        appearances
        .groupby("team")
        .size()
    )

    if not matches_per_team.eq(38).all():
        raise ValueError(
            "Every club must play "
            "38 league matches."
        )

    home_counts = (
        fixtures
        .groupby("home_team")
        .size()
    )

    away_counts = (
        fixtures
        .groupby("away_team")
        .size()
    )

    if not home_counts.eq(19).all():
        raise ValueError(
            "Every club must have "
            "19 home fixtures."
        )

    if not away_counts.eq(19).all():
        raise ValueError(
            "Every club must have "
            "19 away fixtures."
        )

    unordered_pairs = fixtures.apply(
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

    pair_counts = (
        unordered_pairs
        .value_counts()
    )

    if len(pair_counts) != 190:
        raise ValueError(
            "Expected 190 unique club pairs, "
            f"found {len(pair_counts)}."
        )

    if not pair_counts.eq(2).all():
        raise ValueError(
            "Every pair of clubs must meet "
            "exactly twice."
        )

    directed_pairs = set(
        zip(
            fixtures["home_team"],
            fixtures["away_team"],
        )
    )

    for team_a, team_b in pair_counts.index:
        if (
            (team_a, team_b)
            not in directed_pairs
            or
            (team_b, team_a)
            not in directed_pairs
        ):
            raise ValueError(
                "Home/away reversal missing "
                f"for {team_a} / {team_b}."
            )


def print_summary(
    fixtures: pd.DataFrame,
) -> None:
    print()
    print(
        "Premier League 2026-27 "
        "fixture snapshot"
    )
    print("=" * 72)

    print(
        f"Fixtures: "
        f"{len(fixtures)}"
    )

    print(
        "Clubs: "
        f"{len(set(fixtures['home_team']) | set(fixtures['away_team']))}"
    )

    print(
        "Matchdays: "
        f"{fixtures['matchday'].nunique()}"
    )

    print(
        "First fixture date: "
        f"{fixtures['match_date'].min()}"
    )

    print(
        "Final fixture date: "
        f"{fixtures['match_date'].max()}"
    )

    print()
    print("First 10 fixtures:")
    print(
        fixtures[
            [
                "matchday",
                "match_date",
                "kickoff_time",
                "home_team",
                "away_team",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("Validation:")
    print("  20 clubs:                PASS")
    print("  380 fixtures:            PASS")
    print("  38 matchdays:            PASS")
    print("  10 fixtures/matchday:    PASS")
    print("  38 matches/club:         PASS")
    print("  19 home matches/club:    PASS")
    print("  19 away matches/club:    PASS")
    print("  190 opponent pairs:      PASS")
    print("  Home/away reversals:     PASS")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = fetch_source_lines()

    RAW_SNAPSHOT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    fixtures = parse_fixtures(
        lines
    )

    validate_fixture_frame(
        fixtures
    )

    fixtures.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        fixtures
    )

    print()
    print(
        f"Wrote: {OUTPUT_PATH}"
    )

    print(
        f"Raw snapshot: "
        f"{RAW_SNAPSHOT_PATH}"
    )


if __name__ == "__main__":
    main()