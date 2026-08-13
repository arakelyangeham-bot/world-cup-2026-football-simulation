from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta

from fixture_generation.fixture import (
    ScheduledFixture,
)


class RoundRobinFixtureGenerator:
    """
    Generate single- or double-round-robin schedules.

    When `start_date` is provided, every matchday receives a
    calendar date. All fixtures on the same matchday share the
    same date.
    """

    def generate(
        self,
        participants: list[str],
        double_round_robin: bool = True,
        competition_id: str = "competition",
        start_date: (
            str | date | datetime | None
        ) = None,
        days_between_matchdays: int = 7,
    ) -> list[ScheduledFixture]:
        self._validate_participants(
            participants
        )

        if days_between_matchdays < 1:
            raise ValueError(
                "days_between_matchdays must be at "
                "least 1."
            )

        parsed_start_date = self._parse_start_date(
            start_date
        )

        first_leg = self._generate_first_leg(
            participants=participants,
            competition_id=competition_id,
            start_date=parsed_start_date,
            days_between_matchdays=(
                days_between_matchdays
            ),
        )

        if not double_round_robin:
            return first_leg

        first_leg_matchdays = max(
            fixture.matchday
            for fixture in first_leg
        )

        second_leg = [
            ScheduledFixture(
                fixture_id=(
                    f"{competition_id}_md_"
                    f"{fixture.matchday + first_leg_matchdays}_"
                    f"{fixture.away_team}_vs_"
                    f"{fixture.home_team}"
                ),
                matchday=(
                    fixture.matchday
                    + first_leg_matchdays
                ),
                home_team=fixture.away_team,
                away_team=fixture.home_team,
                leg=2,
                match_date=(
                    self._date_for_matchday(
                        start_date=parsed_start_date,
                        matchday=(
                            fixture.matchday
                            + first_leg_matchdays
                        ),
                        days_between_matchdays=(
                            days_between_matchdays
                        ),
                    )
                ),
            )
            for fixture in first_leg
        ]

        return first_leg + second_leg

    def _generate_first_leg(
        self,
        participants: list[str],
        competition_id: str,
        start_date: date | None,
        days_between_matchdays: int,
    ) -> list[ScheduledFixture]:
        rotation: list[str | None] = (
            participants[:]
        )

        if len(rotation) % 2 == 1:
            rotation.append(None)

        team_count = len(rotation)
        round_count = team_count - 1
        matches_per_round = team_count // 2

        fixtures: list[ScheduledFixture] = []

        for round_index in range(
            round_count
        ):
            matchday = round_index + 1

            match_date = (
                self._date_for_matchday(
                    start_date=start_date,
                    matchday=matchday,
                    days_between_matchdays=(
                        days_between_matchdays
                    ),
                )
            )

            for pair_index in range(
                matches_per_round
            ):
                team1 = rotation[pair_index]
                team2 = rotation[
                    team_count - 1 - pair_index
                ]

                if (
                    team1 is None
                    or team2 is None
                ):
                    continue

                if pair_index == 0:
                    if round_index % 2 == 0:
                        home_team = team1
                        away_team = team2
                    else:
                        home_team = team2
                        away_team = team1

                elif (
                    round_index + pair_index
                ) % 2 == 0:
                    home_team = team1
                    away_team = team2

                else:
                    home_team = team2
                    away_team = team1

                fixtures.append(
                    ScheduledFixture(
                        fixture_id=(
                            f"{competition_id}_md_"
                            f"{matchday}_"
                            f"{home_team}_vs_"
                            f"{away_team}"
                        ),
                        matchday=matchday,
                        home_team=home_team,
                        away_team=away_team,
                        leg=1,
                        match_date=match_date,
                    )
                )

            rotation = [
                rotation[0],
                rotation[-1],
                *rotation[1:-1],
            ]

        return fixtures

    @staticmethod
    def _parse_start_date(
        value: (
            str | date | datetime | None
        ),
    ) -> date | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        return date.fromisoformat(value)

    @staticmethod
    def _date_for_matchday(
        start_date: date | None,
        matchday: int,
        days_between_matchdays: int,
    ) -> date | None:
        if start_date is None:
            return None

        return start_date + timedelta(
            days=(
                (matchday - 1)
                * days_between_matchdays
            )
        )

    @staticmethod
    def _validate_participants(
        participants: list[str],
    ) -> None:
        if len(participants) < 2:
            raise ValueError(
                "A round-robin competition requires "
                "at least two teams."
            )

        if len(participants) != len(
            set(participants)
        ):
            duplicates = [
                team
                for team, count
                in Counter(
                    participants
                ).items()
                if count > 1
            ]

            raise ValueError(
                "Round-robin participants contain "
                f"duplicates: {duplicates}"
            )

        if any(
            not team.strip()
            for team in participants
        ):
            raise ValueError(
                "Participant names cannot be empty."
            )