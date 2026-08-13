#test_player_selection_scoring

from __future__ import annotations

import math

import pandas as pd
import pytest

from research.player_intelligence.player_selection_scoring import (
    MINUTES_COMPONENT_COLUMN,
    NORMALIZED_ROLE_RATING_COLUMN,
    RATING_ONLY_SPECIFICATION,
    ROLE_COMPONENT_COLUMN,
    SELECTION_SCORE_COLUMN,
    START_COMPONENT_COLUMN,
    PlayerSelectionSpecification,
    generate_weight_grid,
    normalize_role_ratings,
    rank_candidates,
    score_candidates,
)


def build_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "1",
                "player": "High Rating Reserve",
                "rating_ST": 90.0,
                "start_rate": 0.20,
                "minutes_relative_to_club_max": 0.25,
            },
            {
                "player_id": "2",
                "player": "Regular Starter",
                "rating_ST": 86.0,
                "start_rate": 0.90,
                "minutes_relative_to_club_max": 0.95,
            },
            {
                "player_id": "3",
                "player": "Rotation Player",
                "rating_ST": 82.0,
                "start_rate": 0.50,
                "minutes_relative_to_club_max": 0.55,
            },
            {
                "player_id": "4",
                "player": "Ineligible Player",
                "rating_ST": None,
                "start_rate": 1.00,
                "minutes_relative_to_club_max": 1.00,
            },
        ]
    )


def test_specification_requires_weights_sum_to_one() -> None:
    specification = PlayerSelectionSpecification(
        name="invalid",
        role_rating_weight=0.70,
        start_rate_weight=0.20,
        minutes_weight=0.20,
    )

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        specification.validate()


def test_rating_only_preserves_role_rating_order() -> None:
    candidates = build_candidates()

    ranked = rank_candidates(
        candidates,
        role_rating_column="rating_ST",
        specification=RATING_ONLY_SPECIFICATION,
    )

    assert ranked["player_id"].tolist() == [
        "1",
        "2",
        "3",
    ]


def test_usage_can_promote_regular_starter() -> None:
    candidates = build_candidates()

    specification = PlayerSelectionSpecification(
        name="hybrid",
        role_rating_weight=0.50,
        start_rate_weight=0.25,
        minutes_weight=0.25,
    )

    ranked = rank_candidates(
        candidates,
        role_rating_column="rating_ST",
        specification=specification,
    )

    assert ranked.iloc[0]["player_id"] == "2"
    assert (
        ranked.iloc[0][
            SELECTION_SCORE_COLUMN
        ]
        > ranked.iloc[1][
            SELECTION_SCORE_COLUMN
        ]
    )


def test_ineligible_player_receives_no_score() -> None:
    candidates = build_candidates()

    scored = score_candidates(
        candidates,
        role_rating_column="rating_ST",
        specification=RATING_ONLY_SPECIFICATION,
    )

    ineligible = scored.loc[
        scored["player_id"].eq("4")
    ].iloc[0]

    assert math.isnan(
        ineligible[
            NORMALIZED_ROLE_RATING_COLUMN
        ]
    )

    assert math.isnan(
        ineligible[
            SELECTION_SCORE_COLUMN
        ]
    )


def test_component_sum_equals_selection_score() -> None:
    candidates = build_candidates()

    specification = PlayerSelectionSpecification(
        name="hybrid",
        role_rating_weight=0.60,
        start_rate_weight=0.20,
        minutes_weight=0.20,
    )

    scored = score_candidates(
        candidates,
        role_rating_column="rating_ST",
        specification=specification,
    )

    eligible = scored.loc[
        scored[
            SELECTION_SCORE_COLUMN
        ].notna()
    ]

    expected = (
        eligible[
            ROLE_COMPONENT_COLUMN
        ]
        + eligible[
            START_COMPONENT_COLUMN
        ]
        + eligible[
            MINUTES_COMPONENT_COLUMN
        ]
    )

    assert expected.equals(
        eligible[
            SELECTION_SCORE_COLUMN
        ]
    )


def test_equal_role_ratings_normalize_to_one() -> None:
    values = pd.Series(
        [
            80.0,
            80.0,
            None,
        ]
    )

    normalized = normalize_role_ratings(
        values
    )

    assert normalized.iloc[0] == 1.0
    assert normalized.iloc[1] == 1.0
    assert pd.isna(
        normalized.iloc[2]
    )


def test_weight_grid_respects_simplex() -> None:
    specifications = generate_weight_grid(
        role_weights=(
            0.50,
            0.60,
            0.70,
            0.80,
        ),
        start_weights=(
            0.00,
            0.10,
            0.20,
            0.30,
        ),
    )

    assert specifications

    for specification in specifications:
        specification.validate()

        total = (
            specification.role_rating_weight
            + specification.start_rate_weight
            + specification.minutes_weight
        )

        assert math.isclose(
            total,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_candidate_input_is_not_mutated() -> None:
    candidates = build_candidates()
    original = candidates.copy(
        deep=True
    )

    score_candidates(
        candidates,
        role_rating_column="rating_ST",
        specification=RATING_ONLY_SPECIFICATION,
    )

    pd.testing.assert_frame_equal(
        candidates,
        original,
    )