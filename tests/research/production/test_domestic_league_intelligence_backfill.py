#test_domestic_league_intelligence_backfill
from __future__ import annotations

import pandas as pd
import pytest

from research.production.domestic_league_intelligence_backfill import (
    attach_canonical_player_ids,
    canonicalize_new_player_profiles,
    extend_player_ratings,
    extend_player_registry,
)


def _baseline_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "canonical_player_id": "1",
                "player_id": "1",
                "player": "Existing Player",
                "player_slug": "existing-player",
                "country": "England",
                "country_alpha2": "GB",
                "country_alpha3": "ENG",
                "positions_detailed": "['ST']",
                "eligible_roles": ["ST"],
                "position": "ST",
                "current_team": "Existing FC",
            }
        ]
    )


def _new_profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "2",
                "player": "New Player",
                "player_slug": "new-player",
                "position": "F",
                "positions_detailed": "['ST']",
                "country": "Germany",
                "country_alpha2": "DE",
                "country_alpha3": "GER",
                "current_team": "Promoted FC",
            }
        ]
    )


def test_canonicalize_new_player_profiles() -> None:
    result = canonicalize_new_player_profiles(
        missing_profiles=_new_profile(),
        baseline_registry=_baseline_registry(),
    )

    assert len(result) == 1
    assert result.iloc[0]["player_id"] == "2"
    assert result.iloc[0]["canonical_player_id"] == "2"

    assert result.iloc[0]["eligible_roles"]


def test_extend_player_registry_preserves_baseline() -> None:
    baseline = _baseline_registry()

    new = canonicalize_new_player_profiles(
        missing_profiles=_new_profile(),
        baseline_registry=baseline,
    )

    expanded = extend_player_registry(
        baseline_registry=baseline,
        new_canonical_registry=new,
    )

    assert len(expanded) == 2
    assert expanded["player_id"].nunique() == 2

    assert set(expanded["player_id"]) == {
        "1",
        "2",
    }


def test_attach_canonical_player_ids() -> None:
    registry = pd.DataFrame(
        {
            "player_id": ["10", "20"],
            "canonical_player_id": ["10", "20"],
        }
    )

    features = pd.DataFrame(
        {
            "player_id": ["10", "20"],
            "minutesPlayed": [1000, 900],
        }
    )

    result = attach_canonical_player_ids(
        features=features,
        registry=registry,
    )

    assert list(
        result["canonical_player_id"]
    ) == ["10", "20"]

    assert result[
        "canonical_player_id"
    ].isna().sum() == 0


def test_extend_player_ratings_adds_only_missing_players() -> None:
    membership = pd.DataFrame(
        {
            "player_id": [
                "1",
                "2",
            ]
        }
    )

    baseline = pd.DataFrame(
        {
            "player_id": ["1"],
            "rating_ST": [0.75],
        }
    )

    #
    # The supplementary artifact deliberately contains
    # both players. Player 1 must NOT overwrite the
    # frozen baseline representation.
    #
    supplementary = pd.DataFrame(
        {
            "player_id": [
                "1",
                "2",
            ],
            "rating_ST": [
                -999.0,
                0.60,
            ],
        }
    )

    result = extend_player_ratings(
        membership=membership,
        baseline_ratings=baseline,
        supplementary_ratings=supplementary,
    )

    assert result.requested_missing_players == 1
    assert result.supplementary_players == 2
    assert result.added_players == 1
    assert result.expanded_players == 2

    expanded = (
        result.expanded_ratings
        .set_index("player_id")
    )

    #
    # Critical provenance invariant:
    # existing intelligence survives untouched.
    #
    assert expanded.loc["1", "rating_ST"] == 0.75

    #
    # Genuinely absent intelligence is appended.
    #
    assert expanded.loc["2", "rating_ST"] == 0.60

def test_canonicalize_rejects_baseline_overlap() -> None:
    baseline = _baseline_registry()

    overlapping_profile = pd.DataFrame(
        [
            {
                "player_id": "1",
                "player": "Duplicate Existing Player",
                "player_slug": "duplicate-existing-player",
                "position": "F",
                "positions_detailed": "['ST']",
                "country": "England",
                "country_alpha2": "GB",
                "country_alpha3": "ENG",
                "current_team": "Another FC",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="overlaps the baseline registry",
    ):
        canonicalize_new_player_profiles(
            missing_profiles=overlapping_profile,
            baseline_registry=baseline,
        )


def test_attach_canonical_ids_rejects_missing_identity() -> None:
    registry = pd.DataFrame(
        {
            "player_id": ["10"],
            "canonical_player_id": ["10"],
        }
    )

    features = pd.DataFrame(
        {
            "player_id": [
                "10",
                "20",
            ],
            "minutesPlayed": [
                1000,
                900,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing canonical registry identities",
    ):
        attach_canonical_player_ids(
            features=features,
            registry=registry,
        )


def test_extend_ratings_rejects_incomplete_supplement() -> None:
    membership = pd.DataFrame(
        {
            "player_id": [
                "1",
                "2",
                "3",
            ]
        }
    )

    baseline = pd.DataFrame(
        {
            "player_id": ["1"],
            "rating_ST": [0.75],
        }
    )

    supplementary = pd.DataFrame(
        {
            "player_id": ["2"],
            "rating_ST": [0.60],
        }
    )

    with pytest.raises(
        ValueError,
        match="do not cover all",
    ):
        extend_player_ratings(
            membership=membership,
            baseline_ratings=baseline,
            supplementary_ratings=supplementary,
        )


def test_extend_registry_rejects_duplicate_player_ids() -> None:
    baseline = _baseline_registry()

    duplicate_new = pd.DataFrame(
        [
            {
                "canonical_player_id": "1",
                "player_id": "1",
                "player": "Duplicate",
                "player_slug": "duplicate",
                "country": "England",
                "country_alpha2": "GB",
                "country_alpha3": "ENG",
                "positions_detailed": "['ST']",
                "eligible_roles": ["ST"],
                "position": "ST",
                "current_team": "Duplicate FC",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        extend_player_registry(
            baseline_registry=baseline,
            new_canonical_registry=duplicate_new,
        )