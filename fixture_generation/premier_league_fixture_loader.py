#premier_league_fixture_loader

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fixture_generation import ScheduledFixture


REQUIRED_COLUMNS = {
    "fixture_id",
    "matchday",
    "match_date",
    "home_team",
    "away_team",
}


def load_premier_league_fixtures(
    fixture_path: str | Path,
) -> list[ScheduledFixture]:
    """
    Load a canonical Premier League fixture CSV into
    ScheduledFixture domain objects.

    The CSV is expected to have already passed source-level
    fixture validation. This loader validates the boundary
    between the persisted artifact and the simulation domain.
    """

    fixture_path = Path(fixture_path)

    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Fixture file does not exist: {fixture_path}"
        )

    frame = pd.read_csv(
        fixture_path,
        dtype={
            "fixture_id": str,
            "home_team": str,
            "away_team": str,
        },
    )

    missing_columns = (
        REQUIRED_COLUMNS - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fixture file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError(
            "Fixture file contains no fixtures."
        )

    if frame["fixture_id"].duplicated().any():
        raise ValueError(
            "Fixture file contains duplicate fixture IDs."
        )

    frame["matchday"] = pd.to_numeric(
        frame["matchday"],
        errors="raise",
    ).astype(int)

    parsed_dates = pd.to_datetime(
        frame["match_date"],
        errors="raise",
    )

    fixtures: list[ScheduledFixture] = []

    for index, row in frame.iterrows():
        fixture_id = str(
            row["fixture_id"]
        ).strip()

        home_team = str(
            row["home_team"]
        ).strip()

        away_team = str(
            row["away_team"]
        ).strip()

        matchday = int(
            row["matchday"]
        )

        match_date: date = (
            parsed_dates.loc[index].date()
        )

        if not fixture_id:
            raise ValueError(
                "Fixture contains an empty fixture ID."
            )

        if not home_team or not away_team:
            raise ValueError(
                f"{fixture_id} contains an empty team name."
            )

        if home_team == away_team:
            raise ValueError(
                f"{fixture_id} has the same home and away team."
            )

        if matchday < 1:
            raise ValueError(
                f"{fixture_id} has invalid matchday "
                f"{matchday}."
            )

        # The first 19 rounds constitute the first leg of the
        # double round robin; rounds 20-38 constitute the return
        # leg.
        leg = (
            1
            if matchday <= 19
            else 2
        )

        fixtures.append(
            ScheduledFixture(
                fixture_id=fixture_id,
                matchday=matchday,
                home_team=home_team,
                away_team=away_team,
                leg=leg,
                match_date=match_date,
            )
        )

    return fixtures