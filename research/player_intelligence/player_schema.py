#player_schema.py

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlayerIdentity:
    """
    Stable player identity.

    These fields identify the footballer independently of any
    particular data provider.
    """

    player_id: str
    name: str
    national_team: str

    club: str | None = None

    primary_position: str | None = None
    secondary_positions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayerAvailability:
    """
    Information affecting player selection.

    Initially these values will usually be unknown.

    Future data sources may populate them.
    """

    available: bool = True

    injured: bool = False
    suspended: bool = False

    expected_to_start: bool | None = None

    minutes_fit: float | None = None


@dataclass(frozen=True)
class PlayerRatings:
    """
    Football ability representation.

    These ratings intentionally mirror the team-strength
    dimensions already used by the simulator.
    """

    overall: float

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    recent_form: float | None = None

@dataclass(frozen=True)
class RoleRatings:
    GK: float | None = None
    CB: float | None = None
    FB: float | None = None
    DM: float | None = None
    CM: float | None = None
    AM: float | None = None
    WM: float | None = None
    W: float | None = None
    ST: float | None = None

@dataclass(frozen=True)
class PlayerEvidence:
    minutes_played: float | None = None
    total_weighted_evidence: float | None = None
    evidence_confidence: float | None = None
    competition_count: int | None = None
    season_count: int | None = None
    recency_weight: float | None = None
    sample_quality: str | None = None


@dataclass(frozen=True)
class Player:
    """
    Canonical Player object.

    Every future data source should ultimately map into this
    representation.
    """

    identity: PlayerIdentity

    ratings: PlayerRatings

    availability: PlayerAvailability = field(
        default_factory=PlayerAvailability
    )

    role_ratings: RoleRatings | None = None
    evidence: PlayerEvidence = field(default_factory=PlayerEvidence)
    evidence_history: object | None = None

    


@dataclass(frozen=True)
class Squad:
    """
    Available player pool for one national team.
    """

    national_team: str

    players: tuple[Player, ...]

@dataclass(frozen=True)
class LineupAssignment:
    """
    One player's assignment within a specific lineup.

    This object preserves information already known during lineup
    selection. It does not yet modify player ability or team strength.
    """

    slot: str
    tactical_role: str
    player: Player
    selection_rating: float

    def __post_init__(self) -> None:
        if not self.slot.strip():
            raise ValueError(
                "Lineup assignment slot must not be empty."
            )

        if not self.tactical_role.strip():
            raise ValueError(
                "Lineup assignment tactical role must not be empty."
            )

@dataclass(frozen=True)
class StartingXI:
    """
    Expected starting lineup.

    ``players`` remains the established backward-compatible player
    view used by Version 2 consumers.

    ``assignments`` preserves formation slot and tactical-role context
    for Version 3 consumers. Existing manually constructed StartingXI
    objects may omit assignments.
    """

    national_team: str
    formation: str
    players: tuple[Player, ...]

    assignments: tuple[
        LineupAssignment,
        ...
    ] = ()

    def __post_init__(self) -> None:
        player_ids = tuple(
            str(player.identity.player_id)
            for player in self.players
        )

        if len(player_ids) != len(
            set(player_ids)
        ):
            raise ValueError(
                "Starting XI contains duplicate player IDs."
            )

        if not self.assignments:
            return

        if len(self.assignments) != len(
            self.players
        ):
            raise ValueError(
                "Starting XI assignment count does not match "
                "the player count."
            )

        assignment_player_ids = tuple(
            str(
                assignment
                .player
                .identity
                .player_id
            )
            for assignment in self.assignments
        )

        if assignment_player_ids != player_ids:
            raise ValueError(
                "Starting XI assignments and players are not "
                "aligned in the same order."
            )

        slots = tuple(
            assignment.slot
            for assignment in self.assignments
        )

        if len(slots) != len(set(slots)):
            raise ValueError(
                "Starting XI assignments contain duplicate slots."
            )


@dataclass(frozen=True)
class PlayerDerivedTeamStrength:
    """
    Output of the Player Intelligence layer.

    This object is intentionally shaped to match the existing
    team-strength interface consumed by the production simulator.
    """

    national_team: str

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    poisson_attack: float
    poisson_defense: float

    overall: float | None = None

