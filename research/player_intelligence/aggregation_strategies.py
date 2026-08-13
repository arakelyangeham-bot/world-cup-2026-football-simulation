#aggregation_strategies.py

from __future__ import annotations

from typing import Iterable

from research.player_intelligence.player_representation import (
    PlayerRepresentation,
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


def uniform_mean(
    players: list[PlayerRepresentation],
) -> float:
    """
    Every player contributes equally.
    """
    return _mean(
        player.current_ability
        for player in players
    )


def top_n_mean(
    players: list[PlayerRepresentation],
    n: int = 11,
) -> float:
    """
    Current production philosophy.
    """

    abilities = sorted(
        (
            player.current_ability
            for player in players
        ),
        reverse=True,
    )

    return _mean(abilities[:n])


def starter_plus_depth(
    players: list[PlayerRepresentation],
    starter_weight: float = 1.0,
    bench_weight: float = 0.25,
    starters: int = 11,
) -> float:
    """
    Starters dominate.

    Bench contributes modestly.
    """

    abilities = sorted(
        (
            player.current_ability
            for player in players
        ),
        reverse=True,
    )

    starter_values = abilities[:starters]
    bench_values = abilities[starters:]

    weighted = (
        [starter_weight * value for value in starter_values]
        + [bench_weight * value for value in bench_values]
    )

    total_weight = (
        starter_weight * len(starter_values)
        + bench_weight * len(bench_values)
    )

    if total_weight == 0:
        return 0.0

    return sum(weighted) / total_weight


def star_weighted(
    players: list[PlayerRepresentation],
    gamma: float = 2.0,
) -> float:
    """
    Elite players influence the team disproportionately.
    """

    abilities = [
        max(player.current_ability, 0.0)
        for player in players
    ]

    if not abilities:
        return 0.0

    weights = [
        ability ** gamma
        for ability in abilities
    ]

    total_weight = sum(weights)

    if total_weight == 0:
        return 0.0

    return sum(
        ability * weight
        for ability, weight in zip(
            abilities,
            weights,
        )
    ) / total_weight