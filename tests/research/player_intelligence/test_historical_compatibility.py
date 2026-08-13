#test_historical_compatibility

from __future__ import annotations

from dataclasses import replace

import pytest

from research.player_intelligence.aggregation_functions import (
    ability_power_weighted_mean,
)
from research.player_intelligence.aggregation_strategies import (
    star_weighted,
)
from research.player_intelligence.player_representation import (
    PlayerRepresentation,
)


def _player(
    *,
    player_id: str,
    current_ability: float,
) -> PlayerRepresentation:
    return PlayerRepresentation(
        player_id=player_id,
        player_name=player_id,
        current_ability=current_ability,
        evidence_confidence=1.0,
        total_minutes=1000.0,
        competition_count=1,
        season_count=1,
        latest_season="2024/25",
        recency_share=1.0,
    )


def test_pure_historical_formula_matches_original_adapter() -> None:
    players = [
        _player(
            player_id="player_1",
            current_ability=0.90,
        ),
        _player(
            player_id="player_2",
            current_ability=0.80,
        ),
        _player(
            player_id="player_3",
            current_ability=0.70,
        ),
    ]

    original_result = star_weighted(
        players,
        gamma=2.0,
    )
    pure_result = ability_power_weighted_mean(
        [
            player.current_ability
            for player in players
        ],
        gamma=2.0,
    )

    assert pure_result == pytest.approx(original_result)


@pytest.mark.parametrize(
    "gamma",
    [
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ],
)
def test_pure_historical_formula_matches_original_across_gamma_values(
    gamma: float,
) -> None:
    players = [
        _player(
            player_id="player_1",
            current_ability=0.95,
        ),
        _player(
            player_id="player_2",
            current_ability=0.85,
        ),
        _player(
            player_id="player_3",
            current_ability=0.75,
        ),
        _player(
            player_id="player_4",
            current_ability=0.65,
        ),
    ]

    original_result = star_weighted(
        players,
        gamma=gamma,
    )
    pure_result = ability_power_weighted_mean(
        [
            player.current_ability
            for player in players
        ],
        gamma=gamma,
    )

    assert pure_result == pytest.approx(original_result)


def test_pure_historical_formula_matches_original_with_zero_values() -> None:
    players = [
        _player(
            player_id="player_1",
            current_ability=0.90,
        ),
        _player(
            player_id="player_2",
            current_ability=0.00,
        ),
        _player(
            player_id="player_3",
            current_ability=0.00,
        ),
    ]

    assert ability_power_weighted_mean(
        [
            player.current_ability
            for player in players
        ],
        gamma=2.0,
    ) == pytest.approx(
        star_weighted(
            players,
            gamma=2.0,
        )
    )


def test_pure_historical_formula_matches_original_with_negative_values() -> None:
    players = [
        _player(
            player_id="player_1",
            current_ability=0.90,
        ),
        _player(
            player_id="player_2",
            current_ability=-0.20,
        ),
        _player(
            player_id="player_3",
            current_ability=0.70,
        ),
    ]

    assert ability_power_weighted_mean(
        [
            player.current_ability
            for player in players
        ],
        gamma=2.0,
    ) == pytest.approx(
        star_weighted(
            players,
            gamma=2.0,
        )
    )


def test_pure_historical_formula_matches_original_for_all_zero_weights() -> None:
    players = [
        _player(
            player_id="player_1",
            current_ability=0.00,
        ),
        _player(
            player_id="player_2",
            current_ability=0.00,
        ),
    ]

    assert ability_power_weighted_mean(
        [
            player.current_ability
            for player in players
        ],
        gamma=2.0,
    ) == pytest.approx(
        star_weighted(
            players,
            gamma=2.0,
        )
    )


def test_player_metadata_does_not_affect_historical_aggregation() -> None:
    base_players = [
        _player(
            player_id="player_1",
            current_ability=0.90,
        ),
        _player(
            player_id="player_2",
            current_ability=0.80,
        ),
    ]

    modified_players = [
        replace(
            base_players[0],
            evidence_confidence=0.20,
            total_minutes=50.0,
            competition_count=3,
            season_count=4,
            latest_season="2022/23",
            recency_share=0.10,
        ),
        replace(
            base_players[1],
            evidence_confidence=0.95,
            total_minutes=3000.0,
            competition_count=5,
            season_count=6,
            latest_season="2025/26",
            recency_share=0.90,
        ),
    ]

    assert star_weighted(
        modified_players,
        gamma=2.0,
    ) == pytest.approx(
        star_weighted(
            base_players,
            gamma=2.0,
        )
    )


def test_historical_formula_is_permutation_invariant() -> None:
    players_a = [
        _player(
            player_id="player_1",
            current_ability=0.95,
        ),
        _player(
            player_id="player_2",
            current_ability=0.85,
        ),
        _player(
            player_id="player_3",
            current_ability=0.75,
        ),
    ]
    players_b = [
        players_a[2],
        players_a[0],
        players_a[1],
    ]

    assert star_weighted(
        players_a,
        gamma=2.0,
    ) == pytest.approx(
        star_weighted(
            players_b,
            gamma=2.0,
        )
    )