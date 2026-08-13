#expected_goals_features

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedGoalsFeatures:
    """
    Canonical feature vector supplied to expected-goals models.

    This class intentionally mirrors the production feature schema.
    """

    # Raw team dimensions
    home_attack: float
    home_midfield: float
    home_defense: float
    home_gk: float

    away_attack: float
    away_midfield: float
    away_defense: float
    away_gk: float

    # Engineered differences
    attack_diff: float
    midfield_diff: float
    defense_diff: float
    gk_diff: float

    # Poisson features
    home_poisson_attack: float
    home_poisson_defense: float

    away_poisson_attack: float
    away_poisson_defense: float

    poisson_attack_diff: float
    poisson_defense_diff: float

    # External prior
    fifa_points_diff: float

    schema_version: str = "v2"

def to_dict(self) -> dict[str, float]:
    """
    Preserve compatibility with existing dictionary-based code.
    """

    return {
        field: getattr(self, field)
        for field in self.__dataclass_fields__
        if field != "schema_version"
    }


@classmethod
def from_dict(
    cls,
    values: dict[str, float],
):
    return cls(**values)