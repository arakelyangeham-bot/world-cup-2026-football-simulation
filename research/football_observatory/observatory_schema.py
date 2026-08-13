# observatory_schema.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreMatchObservation:
    home_team: str | None
    away_team: str | None

    home_attack: float
    home_midfield: float
    home_defense: float
    home_gk: float

    away_attack: float
    away_midfield: float
    away_defense: float
    away_gk: float

    attack_diff: float
    midfield_diff: float
    defense_diff: float
    gk_diff: float

    home_poisson_attack: float
    home_poisson_defense: float
    away_poisson_attack: float
    away_poisson_defense: float

    poisson_attack_diff: float
    poisson_defense_diff: float

    home_fifa_points: float
    away_fifa_points: float
    fifa_points_diff: float

    competitive_balance: float | None = None

    @property
    def home_rating_prior(self) -> float:
        """
        Generic external rating prior for the home team.

        Backed temporarily by the legacy home_fifa_points field
        for backward compatibility.
        """
        return self.home_fifa_points

    @property
    def away_rating_prior(self) -> float:
        """
        Generic external rating prior for the away team.

        Backed temporarily by the legacy away_fifa_points field
        for backward compatibility.
        """
        return self.away_fifa_points

    @property
    def rating_prior_diff(self) -> float:
        """
        Home rating prior minus away rating prior.

        Backed temporarily by the legacy fifa_points_diff field
        for backward compatibility.
        """
        return self.fifa_points_diff


@dataclass(frozen=True)
class ObservedMatchOutcome:
    home_score: int
    away_score: int
    result: str

    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score

    @property
    def goal_difference(self) -> int:
        return self.home_score - self.away_score

    @property
    def absolute_goal_difference(self) -> int:
        return abs(self.goal_difference)

    @property
    def is_draw(self) -> bool:
        return self.home_score == self.away_score

    @property
    def is_home_win(self) -> bool:
        return self.home_score > self.away_score

    @property
    def is_away_win(self) -> bool:
        return self.away_score > self.home_score

    @property
    def is_one_goal_match(self) -> bool:
        return self.absolute_goal_difference == 1

    @property
    def is_clean_sheet(self) -> bool:
        return self.home_score == 0 or self.away_score == 0

    @property
    def both_teams_scored(self) -> bool:
        return self.home_score > 0 and self.away_score > 0

    @property
    def is_high_scoring(self) -> bool:
        return self.total_goals >= 5

    @property
    def is_blowout(self) -> bool:
        return self.absolute_goal_difference >= 3

    @property
    def scoreline(self) -> str:
        return f"{self.home_score}-{self.away_score}"


@dataclass(frozen=True)
class MatchObservation:
    prematch: PreMatchObservation
    outcome: ObservedMatchOutcome
    events: Any | None = None
    derived_prematch: dict[str, float] | None = None

def _resolve_rating_prior_value(
    row,
    canonical_column: str,
    legacy_column: str,
) -> float:
    """
    Resolve one generic rating-prior feature.

    Canonical rating_prior columns take precedence. Legacy FIFA-point
    columns remain supported while the national pipeline is migrated.
    """

    canonical_present = (
        canonical_column in row
        and row[canonical_column] is not None
    )

    legacy_present = (
        legacy_column in row
        and row[legacy_column] is not None
    )

    if canonical_present:
        canonical_value = float(
            row[canonical_column]
        )

        if legacy_present:
            legacy_value = float(
                row[legacy_column]
            )

            if abs(
                canonical_value - legacy_value
            ) > 1e-12:
                raise ValueError(
                    "Conflicting canonical and legacy "
                    "rating-prior values: "
                    f"{canonical_column}={canonical_value}, "
                    f"{legacy_column}={legacy_value}."
                )

        return canonical_value

    if legacy_present:
        return float(
            row[legacy_column]
        )

    raise KeyError(
        "Missing rating-prior feature. Expected either "
        f"{canonical_column!r} or {legacy_column!r}."
    )

def prematch_from_row(row) -> PreMatchObservation:
    return PreMatchObservation(
        home_team=row.get("home_team"),
        away_team=row.get("away_team"),

        home_attack=float(row["home_attack"]),
        home_midfield=float(row["home_midfield"]),
        home_defense=float(row["home_defense"]),
        home_gk=float(row["home_gk"]),

        away_attack=float(row["away_attack"]),
        away_midfield=float(row["away_midfield"]),
        away_defense=float(row["away_defense"]),
        away_gk=float(row["away_gk"]),

        attack_diff=float(row["attack_diff"]),
        midfield_diff=float(row["midfield_diff"]),
        defense_diff=float(row["defense_diff"]),
        gk_diff=float(row["gk_diff"]),

        home_poisson_attack=float(row["home_poisson_attack"]),
        home_poisson_defense=float(row["home_poisson_defense"]),
        away_poisson_attack=float(row["away_poisson_attack"]),
        away_poisson_defense=float(row["away_poisson_defense"]),

        poisson_attack_diff=float(row["poisson_attack_diff"]),
        poisson_defense_diff=float(row["poisson_defense_diff"]),

        home_fifa_points=(
            _resolve_rating_prior_value(
                row=row,
                canonical_column=(
                    "home_rating_prior"
                ),
                legacy_column=(
                    "home_fifa_points"
                ),
            )
        ),
        away_fifa_points=(
            _resolve_rating_prior_value(
                row=row,
                canonical_column=(
                    "away_rating_prior"
                ),
                legacy_column=(
                    "away_fifa_points"
                ),
            )
        ),
        fifa_points_diff=(
            _resolve_rating_prior_value(
                row=row,
                canonical_column=(
                    "rating_prior_diff"
                ),
                legacy_column=(
                    "fifa_points_diff"
                ),
            )
        ),
        competitive_balance=(
            float(row["competitive_balance"])
            if "competitive_balance" in row
            else None
        ),
    )


def outcome_from_row(row) -> ObservedMatchOutcome:
    return ObservedMatchOutcome(
        home_score=int(row["home_score"]),
        away_score=int(row["away_score"]),
        result=str(row["result"]),
    )


def match_observation_from_row(row) -> MatchObservation:
    return MatchObservation(
        prematch=prematch_from_row(row),
        outcome=outcome_from_row(row),
        events=None,
    )