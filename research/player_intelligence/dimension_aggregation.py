#dimension_aggregation.py

from __future__ import annotations

from research.player_intelligence.aggregation_strategies import (
    star_weighted,
    starter_plus_depth,
    top_n_mean,
)
from research.player_intelligence.player_representation import (
    PlayerRepresentation,
)


def best_player(
    players: list[PlayerRepresentation],
) -> float:
    if not players:
        return 0.0

    return max(player.current_ability for player in players)


def aggregate_dimension(
    players: list[PlayerRepresentation],
    strategy: str,
) -> float:
    if strategy == "star_weighted":
        return star_weighted(players)

    if strategy == "starter_plus_depth":
        return starter_plus_depth(players)

    if strategy == "top_11_mean":
        return top_n_mean(players, n=11)

    if strategy == "top_5_mean":
        return top_n_mean(players, n=5)

    if strategy == "best_player":
        return best_player(players)

    raise ValueError(f"Unknown dimension aggregation strategy: {strategy}")


DEFAULT_DIMENSION_STRATEGIES = {
    "attack": "star_weighted",
    "midfield": "starter_plus_depth",
    "defense": "top_11_mean",
    "goalkeeper": "best_player",
}


def aggregate_team_dimensions(
    players: list[PlayerRepresentation],
    strategies: dict[str, str] | None = None,
) -> dict[str, float]:
    selected = strategies or DEFAULT_DIMENSION_STRATEGIES

    return {
        dimension: aggregate_dimension(players, strategy)
        for dimension, strategy in selected.items()
    }