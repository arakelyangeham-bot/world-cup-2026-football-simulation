#validate_team_representation_v2

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from statistics import mean

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.role_projection import (
    project_attack,
    project_defense,
    project_goalkeeper,
    project_midfield,
)
from research.player_intelligence.roster_builder import RosterBuilder
from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_squad,
)


ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LegacyNumericalRepresentation:
    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    attack_depth: float
    midfield_depth: float
    defense_depth: float

    squad_quality: float
    evidence_score: float


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _top_n_mean(values: list[float], n: int) -> float:
    if not values:
        return 0.0

    return _mean(sorted(values, reverse=True)[:n])


def reproduce_legacy_numerical_representation(
    players,
) -> LegacyNumericalRepresentation:
    """
    Independently reproduce the Team Representation Version 1 formulas.

    This function intentionally does not call the production representation
    builder. It acts as a reference implementation for equivalence testing.
    """

    attack_values = [
        project_attack(player.role_ratings)
        for player in players
    ]
    midfield_values = [
        project_midfield(player.role_ratings)
        for player in players
    ]
    defense_values = [
        project_defense(player.role_ratings)
        for player in players
    ]
    goalkeeper_values = [
        project_goalkeeper(player.role_ratings)
        for player in players
    ]
    overall_values = [
        player.ratings.overall
        for player in players
    ]

    evidence_values = [
        1.0
        for player in players
        if player.ratings.overall > 0
    ]

    return LegacyNumericalRepresentation(
        attack=_top_n_mean(attack_values, 5),
        midfield=_top_n_mean(midfield_values, 5),
        defense=_top_n_mean(defense_values, 5),
        goalkeeper=(
            max(goalkeeper_values)
            if goalkeeper_values
            else 0.0
        ),
        attack_depth=_mean(attack_values),
        midfield_depth=_mean(midfield_values),
        defense_depth=_mean(defense_values),
        squad_quality=_mean(overall_values),
        evidence_score=(
            len(evidence_values) / len(players)
            if players
            else 0.0
        ),
    )


def assert_numerical_equivalence(
    team: str,
    actual: TeamRepresentation,
    expected: LegacyNumericalRepresentation,
) -> None:
    fields = (
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "attack_depth",
        "midfield_depth",
        "defense_depth",
        "squad_quality",
        "evidence_score",
    )

    for field in fields:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)

        if not isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=ABS_TOLERANCE,
        ):
            raise AssertionError(
                f"{team}: numerical mismatch for {field}. "
                f"Expected {expected_value}, received {actual_value}."
            )


def assert_provenance_fields(
    team: str,
    representation: TeamRepresentation,
    players,
) -> None:
    expected_available_count = sum(
        1
        for player in players
        if player.availability.available
    )

    if representation.representation_type != "full_squad":
        raise AssertionError(
            f"{team}: expected representation_type='full_squad', "
            f"received {representation.representation_type!r}."
        )

    if representation.aggregation_profile != "legacy_top_5":
        raise AssertionError(
            f"{team}: expected aggregation_profile='legacy_top_5', "
            f"received {representation.aggregation_profile!r}."
        )

    if representation.player_count != len(players):
        raise AssertionError(
            f"{team}: expected player_count={len(players)}, "
            f"received {representation.player_count}."
        )

    if representation.available_player_count != expected_available_count:
        raise AssertionError(
            f"{team}: expected available_player_count="
            f"{expected_available_count}, received "
            f"{representation.available_player_count}."
        )

    if representation.available_player_count > representation.player_count:
        raise AssertionError(
            f"{team}: available player count exceeds total player count."
        )

    if not 0.0 <= representation.evidence_score <= 1.0:
        raise AssertionError(
            f"{team}: evidence_score must be between 0 and 1, "
            f"received {representation.evidence_score}."
        )


def print_summary(
    representations: list[TeamRepresentation],
) -> None:
    player_counts = [
        representation.player_count
        for representation in representations
    ]
    available_counts = [
        representation.available_player_count
        for representation in representations
    ]
    evidence_scores = [
        representation.evidence_score
        for representation in representations
    ]

    representation_type_counts: dict[str, int] = {}
    aggregation_profile_counts: dict[str, int] = {}

    for representation in representations:
        representation_type_counts[
            representation.representation_type
        ] = (
            representation_type_counts.get(
                representation.representation_type,
                0,
            )
            + 1
        )

        aggregation_profile_counts[
            representation.aggregation_profile
        ] = (
            aggregation_profile_counts.get(
                representation.aggregation_profile,
                0,
            )
            + 1
        )

    print("Team Representation V2 Validation")
    print("=================================")
    print()
    print(f"Teams processed: {len(representations)}")

    print()
    print("Representation types")
    print("--------------------")
    for name, count in sorted(representation_type_counts.items()):
        print(f"{name}: {count}")

    print()
    print("Aggregation profiles")
    print("--------------------")
    for name, count in sorted(aggregation_profile_counts.items()):
        print(f"{name}: {count}")

    print()
    print("Player counts")
    print("-------------")
    print(f"Minimum: {min(player_counts)}")
    print(f"Maximum: {max(player_counts)}")
    print(f"Average: {mean(player_counts):.2f}")

    print()
    print("Available player counts")
    print("-----------------------")
    print(f"Minimum: {min(available_counts)}")
    print(f"Maximum: {max(available_counts)}")
    print(f"Average: {mean(available_counts):.2f}")

    print()
    print("Evidence score")
    print("--------------")
    print(f"Minimum: {min(evidence_scores):.4f}")
    print(f"Maximum: {max(evidence_scores):.4f}")
    print(f"Average: {mean(evidence_scores):.4f}")

    print()
    print("Numerical equivalence: PASS")
    print("Provenance validation: PASS")
    print()
    print("OVERALL RESULT: PASS")


def main() -> None:
    evidence_repository = PlayerEvidenceRepository()

    player_repository = PlayerRepository(
        evidence_repository=evidence_repository,
    )

    roster_builder = RosterBuilder(
        repository=player_repository,
    )

    representations: list[TeamRepresentation] = []

    for team in roster_builder.list_teams():
        squad = roster_builder.get_squad(team)

        if not squad.players:
            continue

        actual = build_team_representation_from_squad(squad)

        expected = reproduce_legacy_numerical_representation(
            squad.players
        )

        assert_numerical_equivalence(
            team=team,
            actual=actual,
            expected=expected,
        )

        assert_provenance_fields(
            team=team,
            representation=actual,
            players=squad.players,
        )

        representations.append(actual)

    if not representations:
        raise RuntimeError(
            "No team representations were produced. "
            "Check the player evidence and roster repositories."
        )

    print_summary(representations)


if __name__ == "__main__":
    main()