#validate_team_repository_entry_v2

from __future__ import annotations

from math import isclose

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.roster_builder import RosterBuilder
from research.player_intelligence.team_representation_builder import (
    build_team_representation_from_squad,
)
from research.player_intelligence.team_repository_builder import (
    project_representation_to_repository_entry,
    repository_entry_to_dict,
)


ABS_TOLERANCE = 1e-12


def assert_equal_float(
    team: str,
    field: str,
    actual: float,
    expected: float,
) -> None:
    if not isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=ABS_TOLERANCE,
    ):
        raise AssertionError(
            f"{team}: mismatch for {field}. "
            f"Expected {expected}, received {actual}."
        )


def main() -> None:
    evidence_repository = PlayerEvidenceRepository()

    player_repository = PlayerRepository(
        evidence_repository=evidence_repository,
    )

    roster_builder = RosterBuilder(
        repository=player_repository,
    )

    teams_processed = 0

    for team in roster_builder.list_teams():
        squad = roster_builder.get_squad(team)

        if not squad.players:
            continue

        representation = build_team_representation_from_squad(squad)

        entry = project_representation_to_repository_entry(
            representation=representation,
            fifa_points=None,
        )

        entry_dict = repository_entry_to_dict(entry)

        assert_equal_float(
            team,
            "attack",
            entry.attack,
            representation.attack,
        )
        assert_equal_float(
            team,
            "midfield",
            entry.midfield,
            representation.midfield,
        )
        assert_equal_float(
            team,
            "defense",
            entry.defense,
            representation.defense,
        )
        assert_equal_float(
            team,
            "gk",
            entry.gk,
            representation.goalkeeper,
        )
        assert_equal_float(
            team,
            "poisson_attack",
            entry.poisson_attack,
            representation.attack,
        )
        assert_equal_float(
            team,
            "poisson_defense",
            entry.poisson_defense,
            representation.defense,
        )

        if entry.representation_type != representation.representation_type:
            raise AssertionError(
                f"{team}: representation_type was not preserved."
            )

        if entry.aggregation_profile != representation.aggregation_profile:
            raise AssertionError(
                f"{team}: aggregation_profile was not preserved."
            )

        if entry.player_count != representation.player_count:
            raise AssertionError(
                f"{team}: player_count was not preserved."
            )

        if (
            entry.available_player_count
            != representation.available_player_count
        ):
            raise AssertionError(
                f"{team}: available_player_count was not preserved."
            )

        expected_keys = {
            "attack",
            "midfield",
            "defense",
            "gk",
            "poisson_attack",
            "poisson_defense",
            "representation_type",
            "aggregation_profile",
            "player_count",
            "available_player_count",
            "fifa_points",
        }

        if set(entry_dict) != expected_keys:
            missing = expected_keys - set(entry_dict)
            unexpected = set(entry_dict) - expected_keys

            raise AssertionError(
                f"{team}: repository dictionary schema mismatch. "
                f"Missing={missing}, unexpected={unexpected}."
            )

        teams_processed += 1

    if teams_processed == 0:
        raise RuntimeError(
            "No repository entries were produced."
        )

    print("Team Repository Entry V2 Validation")
    print("===================================")
    print()
    print(f"Teams processed: {teams_processed}")
    print()
    print("Strength projection: PASS")
    print("Provenance propagation: PASS")
    print("Dictionary schema: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()