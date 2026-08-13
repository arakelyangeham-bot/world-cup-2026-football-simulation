from __future__ import annotations

from dataclasses import dataclass


RuleKey = tuple[str, int]


@dataclass(frozen=True)
class DomesticLeagueRules:
    competition_key: str
    season_start_year: int
    team_count: int
    matches_per_team: int
    home_matches_per_team: int
    away_matches_per_team: int

    @property
    def completed_match_count(self) -> int:
        return (
            self.team_count
            * self.matches_per_team
            // 2
        )

    @property
    def unique_pairing_count(self) -> int:
        return (
            self.team_count
            * (self.team_count - 1)
            // 2
        )


def build_rules(
    competition_key: str,
    season_start_year: int,
    team_count: int,
) -> DomesticLeagueRules:
    """
    Construct double round-robin rules from the number of clubs.
    """

    if team_count < 2:
        raise ValueError(
            "A domestic league must contain at least two clubs."
        )

    matches_per_team = 2 * (team_count - 1)
    home_matches_per_team = team_count - 1
    away_matches_per_team = team_count - 1

    return DomesticLeagueRules(
        competition_key=competition_key,
        season_start_year=season_start_year,
        team_count=team_count,
        matches_per_team=matches_per_team,
        home_matches_per_team=home_matches_per_team,
        away_matches_per_team=away_matches_per_team,
    )


DOMESTIC_LEAGUE_RULES: dict[
    RuleKey,
    DomesticLeagueRules,
] = {}


def register_seasons(
    competition_key: str,
    season_start_years: tuple[int, ...],
    team_count: int,
) -> None:
    """
    Register the same league structure for several explicit seasons.
    """

    for season_start_year in season_start_years:
        key = (
            competition_key,
            season_start_year,
        )

        if key in DOMESTIC_LEAGUE_RULES:
            raise ValueError(
                "Duplicate domestic-league rule registration for "
                f"{competition_key!r}, season "
                f"{season_start_year}."
            )

        DOMESTIC_LEAGUE_RULES[key] = build_rules(
            competition_key=competition_key,
            season_start_year=season_start_year,
            team_count=team_count,
        )


register_seasons(
    competition_key="premier_league",
    season_start_years=(
        2021,
        2022,
        2023,
        2024,
        2025,
    ),
    team_count=20,
)

register_seasons(
    competition_key="la_liga",
    season_start_years=(
        2021,
        2022,
        2023,
        2024,
        2025,
    ),
    team_count=20,
)

register_seasons(
    competition_key="serie_a",
    season_start_years=(
        2021,
        2022,
        2023,
        2024,
        2025,
    ),
    team_count=20,
)

register_seasons(
    competition_key="bundesliga",
    season_start_years=(
        2021,
        2022,
        2023,
        2024,
        2025,
    ),
    team_count=18,
)

register_seasons(
    competition_key="ligue_1",
    season_start_years=(
        2021,
        2022,
    ),
    team_count=20,
)

register_seasons(
    competition_key="ligue_1",
    season_start_years=(
        2023,
        2024,
        2025,
    ),
    team_count=18,
)


def get_domestic_league_rules(
    competition_key: str,
    season_start_year: int,
) -> DomesticLeagueRules:
    key = (
        competition_key,
        season_start_year,
    )

    if key not in DOMESTIC_LEAGUE_RULES:
        registered_seasons = sorted(
            year
            for (
                registered_competition,
                year,
            ) in DOMESTIC_LEAGUE_RULES
            if registered_competition
            == competition_key
        )

        known_competitions = sorted(
            {
                registered_competition
                for (
                    registered_competition,
                    _,
                ) in DOMESTIC_LEAGUE_RULES
            }
        )

        if competition_key not in known_competitions:
            raise KeyError(
                "No domestic-league rules are registered for "
                f"{competition_key!r}. Known competitions: "
                f"{known_competitions}"
            )

        raise KeyError(
            "No domestic-league rules are registered for "
            f"{competition_key!r}, season "
            f"{season_start_year}. Registered seasons: "
            f"{registered_seasons}"
        )

    return DOMESTIC_LEAGUE_RULES[key]