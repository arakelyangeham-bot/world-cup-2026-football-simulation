#test_build_player_intelligence_team_repository

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from research.player_intelligence.player_schema import (
    Player,
    PlayerIdentity,
    PlayerRatings,
    RoleRatings,
    Squad,
)
from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
)
from scripts.build_player_intelligence_team_repository import (
    build_repository_dataframe,
    resolve_representation_builder,
)


class FakeRosterBuilder:
    def __init__(
        self,
        squads: dict[str, Squad],
    ) -> None:
        self.squads = squads

    def list_teams(self) -> list[str]:
        return list(
            self.squads
        )

    def get_squad(
        self,
        team: str,
    ) -> Squad:
        return self.squads[
            team
        ]


def make_player(
    player_id: str,
    *,
    national_team: str,
) -> Player:
    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=f"Player {player_id}",
            national_team=national_team,
        ),
        ratings=PlayerRatings(
            overall=0.80,
            attack=0.75,
            midfield=0.80,
            defense=0.70,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            CM=0.80,
        ),
    )


def make_squad(
    national_team: str,
) -> Squad:
    return Squad(
        national_team=national_team,
        players=(
            make_player(
                "1",
                national_team=national_team,
            ),
            make_player(
                "2",
                national_team=national_team,
            ),
        ),
    )


def fake_representation_builder(
    squad: Squad,
) -> TeamRepresentation:
    return TeamRepresentation(
        national_team=(
            squad.national_team
        ),
        representation_type="test",
        aggregation_profile="test_profile",
        attack=0.80,
        midfield=0.70,
        defense=0.60,
        goalkeeper=0.50,
        attack_depth=0.40,
        midfield_depth=0.30,
        defense_depth=0.20,
        squad_quality=0.75,
        evidence_score=1.0,
        player_count=len(
            squad.players
        ),
        available_player_count=len(
            squad.players
        ),
    )


def test_default_policy_resolves() -> None:
    builder = (
        resolve_representation_builder(
            "full_squad_legacy"
        )
    )

    assert callable(
        builder
    )


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown representation policy",
    ):
        resolve_representation_builder(
            "not_a_policy"
        )


def test_repository_builder_uses_injected_policy() -> None:
    roster_builder = FakeRosterBuilder(
        {
            "Test Team":
                make_squad(
                    "Test Team"
                ),
        }
    )

    repository = (
        build_repository_dataframe(
            roster_builder=roster_builder,
            representation_builder=(
                fake_representation_builder
            ),
            fifa_lookup={
                "Test Team": 1500.0,
            },
        )
    )

    assert len(repository) == 1

    row = repository.iloc[0]

    assert row["nation"] == "Test Team"
    assert row["att_composite"] == pytest.approx(
        0.80
    )
    assert row["mid_composite"] == pytest.approx(
        0.70
    )
    assert row["def_composite"] == pytest.approx(
        0.60
    )
    assert row["gk_composite"] == pytest.approx(
        0.50
    )

    assert row[
        "poisson_attack_adj"
    ] == pytest.approx(
        0.80
    )

    assert row[
        "poisson_defense_adj"
    ] == pytest.approx(
        0.60
    )

    assert row[
        "representation_type"
    ] == "test"

    assert row[
        "aggregation_profile"
    ] == "test_profile"

    assert row["player_count"] == 2
    assert row[
        "available_player_count"
    ] == 2

    assert row[
        "fifa_points"
    ] == pytest.approx(
        1500.0
    )


def test_team_identity_mismatch_is_rejected() -> None:
    roster_builder = FakeRosterBuilder(
        {
            "Test Team":
                make_squad(
                    "Test Team"
                ),
        }
    )

    def mismatched_builder(
        squad: Squad,
    ) -> TeamRepresentation:
        return replace(
            fake_representation_builder(
                squad
            ),
            national_team="Other Team",
        )

    with pytest.raises(
        ValueError,
        match="does not match the source squad",
    ):
        build_repository_dataframe(
            roster_builder=roster_builder,
            representation_builder=(
                mismatched_builder
            ),
            fifa_lookup={
                "Test Team": 1500.0,
            },
        )