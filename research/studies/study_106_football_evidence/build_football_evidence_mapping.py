#build_football_evidence_mapping

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_106_football_evidence"
    / "study_106a"
)

OBSERVABLE_CATALOG_PATH = (
    OUTPUT_DIRECTORY
    / "football_observable_catalog.csv"
)

EVIDENCE_MATRIX_PATH = (
    OUTPUT_DIRECTORY
    / "hypothesis_evidence_matrix.csv"
)

DATA_CAPABILITY_PATH = (
    OUTPUT_DIRECTORY
    / "current_data_capability_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_106a_metadata.json"
)

PROTOCOL_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_106A_PROTOCOL.md"
)

BACKLOG_PATH = (
    OUTPUT_DIRECTORY
    / "FOOTBALL_RESEARCH_BACKLOG.md"
)

class EvidenceAvailability(str, Enum):
    """
    Whether the current project pipeline can construct an observable.
    """

    CURRENTLY_AVAILABLE = "currently_available"
    PARTIALLY_AVAILABLE = "partially_available"
    REQUIRES_NEW_ENDPOINT = "requires_new_endpoint"
    NOT_OBSERVABLE = "not_observable"


class EvidenceDirectness(str, Enum):
    """
    Distance between an observable and the underlying football claim.
    """

    DIRECT = "direct"
    STRONG_PROXY = "strong_proxy"
    MODERATE_PROXY = "moderate_proxy"
    WEAK_PROXY = "weak_proxy"
    SPECULATIVE = "speculative"


class MeasurementLevel(str, Enum):
    PLAYER_MATCH = "player_match"
    TEAM_MATCH = "team_match"
    EVENT = "event"
    SPATIAL_EVENT = "spatial_event"
    DEFENSIVE_SEQUENCE = "defensive_sequence"
    SEASON = "season"
    HUMAN_ANNOTATION = "human_annotation"


class ValidationPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"

@dataclass(frozen=True)
class ObservableFootballEvidence:
    observable_id: str
    name: str
    description: str

    availability: EvidenceAvailability
    directness: EvidenceDirectness
    measurement_level: MeasurementLevel

    current_source: str
    required_fields: tuple[str, ...]

    spatial_information_required: bool
    player_identity_required: bool
    sequence_context_required: bool

    notes: str

    def __post_init__(self) -> None:
        if not self.observable_id.strip():
            raise ValueError(
                "Observable ID must not be empty."
            )

        if not self.name.strip():
            raise ValueError(
                "Observable name must not be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Observable description must not be empty."
            )

        if not self.current_source.strip():
            raise ValueError(
                "Observable source description must not be empty."
            )

        if not self.notes.strip():
            raise ValueError(
                "Observable notes must not be empty."
            )


@dataclass(frozen=True)
class HypothesisEvidenceMapping:
    hypothesis_id: str
    observable_id: str

    predicted_tendency: str
    discriminating_value: str

    validation_priority: ValidationPriority

    supports_shared_screen_if: str
    supports_side_specific_if: str

    current_testability: EvidenceAvailability

    confounders: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError(
                "Hypothesis ID must not be empty."
            )

        if not self.observable_id.strip():
            raise ValueError(
                "Observable ID must not be empty."
            )

        if not self.predicted_tendency.strip():
            raise ValueError(
                "Predicted tendency must not be empty."
            )

        if not self.discriminating_value.strip():
            raise ValueError(
                "Discriminating value must not be empty."
            )

        if not self.supports_shared_screen_if.strip():
            raise ValueError(
                "Shared-screen criterion must not be empty."
            )

        if not self.supports_side_specific_if.strip():
            raise ValueError(
                "Side-specific criterion must not be empty."
            )

OBSERVABLES = (
    ObservableFootballEvidence(
        observable_id="player_interceptions_per90",
        name="Player interceptions per 90",
        description=(
            "Interception frequency for each defensive midfielder."
        ),
        availability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.WEAK_PROXY
        ),
        measurement_level=(
            MeasurementLevel.SEASON
        ),
        current_source=(
            "Player model features and team aggregation pipeline."
        ),
        required_fields=(
            "player_id",
            "position",
            "minutesPlayed",
            "interceptions_per90",
        ),
        spatial_information_required=False,
        player_identity_required=True,
        sequence_context_required=False,
        notes=(
            "Measures defensive activity volume but does not identify "
            "which center-back, side, or corridor was protected."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="player_ball_recoveries_per90",
        name="Player ball recoveries per 90",
        description=(
            "Ball-recovery frequency for each member of the "
            "double pivot."
        ),
        availability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.WEAK_PROXY
        ),
        measurement_level=(
            MeasurementLevel.SEASON
        ),
        current_source=(
            "Player model features and team aggregation pipeline."
        ),
        required_fields=(
            "player_id",
            "position",
            "minutesPlayed",
            "ballRecovery_per90",
        ),
        spatial_information_required=False,
        player_identity_required=True,
        sequence_context_required=False,
        notes=(
            "Can compare activity balance between the two DMs, "
            "but not lateral specialization."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="player_tackles_per90",
        name="Player tackles per 90",
        description=(
            "Tackle frequency for each defensive midfielder."
        ),
        availability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.WEAK_PROXY
        ),
        measurement_level=(
            MeasurementLevel.SEASON
        ),
        current_source=(
            "Player model features and team aggregation pipeline."
        ),
        required_fields=(
            "player_id",
            "position",
            "minutesPlayed",
            "tackles_per90",
        ),
        spatial_information_required=False,
        player_identity_required=True,
        sequence_context_required=False,
        notes=(
            "Defensive workload is available, but role responsibility "
            "cannot be reconstructed from volume alone."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="dm_defensive_action_balance",
        name="Double-pivot defensive-action balance",
        description=(
            "Difference between the two DMs in interceptions, "
            "recoveries, tackles, and duels."
        ),
        availability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.MODERATE_PROXY
        ),
        measurement_level=(
            MeasurementLevel.SEASON
        ),
        current_source=(
            "Derived from existing player per-90 features."
        ),
        required_fields=(
            "interceptions_per90",
            "ballRecovery_per90",
            "tackles_per90",
            "totalDuelsWon_per90",
        ),
        spatial_information_required=False,
        player_identity_required=True,
        sequence_context_required=False,
        notes=(
            "Balanced activity may be consistent with shared "
            "responsibility, but player quality and role asymmetry "
            "are major confounders."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="team_defensive_match_statistics",
        name="Team defensive match statistics",
        description=(
            "Team-level interceptions, tackles, recoveries, duels, "
            "clearances, and goals conceded for one match."
        ),
        availability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.WEAK_PROXY
        ),
        measurement_level=(
            MeasurementLevel.TEAM_MATCH
        ),
        current_source=(
            "Sofascore full-match team-statistics endpoint."
        ),
        required_fields=(
            "event_id",
            "team_id",
            "interceptions",
            "tackles",
            "ballRecovery",
            "goals_conceded",
        ),
        spatial_information_required=False,
        player_identity_required=False,
        sequence_context_required=False,
        notes=(
            "Useful for outcomes and context, but cannot distinguish "
            "the two double-pivot hypotheses directly."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="dm_action_corridor_distribution",
        name="DM defensive-action corridor distribution",
        description=(
            "Distribution of each DM's defensive actions across "
            "left, central, and right tactical corridors."
        ),
        availability=(
            EvidenceAvailability.REQUIRES_NEW_ENDPOINT
        ),
        directness=(
            EvidenceDirectness.STRONG_PROXY
        ),
        measurement_level=(
            MeasurementLevel.SPATIAL_EVENT
        ),
        current_source=(
            "Not present in the uploaded match-statistics or "
            "aggregation pipeline."
        ),
        required_fields=(
            "event_id",
            "player_id",
            "action_type",
            "x",
            "y",
            "timestamp",
        ),
        spatial_information_required=True,
        player_identity_required=True,
        sequence_context_required=False,
        notes=(
            "This is the clearest feasible discriminator between "
            "shared and side-specific defensive responsibility."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="dm_to_cb_spatial_proximity",
        name="DM-to-CB spatial proximity",
        description=(
            "Average defensive-phase distance from each DM to "
            "each center-back."
        ),
        availability=(
            EvidenceAvailability.NOT_OBSERVABLE
        ),
        directness=(
            EvidenceDirectness.DIRECT
        ),
        measurement_level=(
            MeasurementLevel.DEFENSIVE_SEQUENCE
        ),
        current_source=(
            "Requires synchronized player-location or tracking data."
        ),
        required_fields=(
            "event_id",
            "timestamp",
            "player_id",
            "x",
            "y",
            "phase_of_play",
        ),
        spatial_information_required=True,
        player_identity_required=True,
        sequence_context_required=True,
        notes=(
            "Would directly distinguish shared proximity from "
            "side-specific pairing, but is not supported by the "
            "current Sofascore pipeline."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="cb_step_out_cover_event",
        name="Center-back step-out cover event",
        description=(
            "Which DM covers the space behind a center-back who "
            "steps forward to engage an opponent."
        ),
        availability=(
            EvidenceAvailability.NOT_OBSERVABLE
        ),
        directness=(
            EvidenceDirectness.DIRECT
        ),
        measurement_level=(
            MeasurementLevel.DEFENSIVE_SEQUENCE
        ),
        current_source=(
            "Would require event sequencing, player locations, "
            "and tactical phase annotation."
        ),
        required_fields=(
            "event_id",
            "timestamp",
            "player_locations",
            "ball_location",
            "defensive_phase",
        ),
        spatial_information_required=True,
        player_identity_required=True,
        sequence_context_required=True,
        notes=(
            "One of the strongest conceptual tests, but unavailable "
            "from aggregate event counts."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="manual_sequence_annotation",
        name="Human-annotated defensive sequences",
        description=(
            "Expert-coded observations of which DM assumes coverage "
            "responsibility during selected defensive sequences."
        ),
        availability=(
            EvidenceAvailability.PARTIALLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.DIRECT
        ),
        measurement_level=(
            MeasurementLevel.HUMAN_ANNOTATION
        ),
        current_source=(
            "Requires match video and a new annotation protocol."
        ),
        required_fields=(
            "event_id",
            "sequence_id",
            "annotator_id",
            "dm_slot",
            "cb_slot",
            "responsibility_label",
        ),
        spatial_information_required=True,
        player_identity_required=True,
        sequence_context_required=True,
        notes=(
            "Technically feasible without tracking data, but costly "
            "and vulnerable to annotator disagreement."
        ),
    ),

    ObservableFootballEvidence(
        observable_id="goals_conceded",
        name="Goals conceded",
        description=(
            "Goals allowed by the team during matches using a "
            "double-pivot formation."
        ),
        availability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        directness=(
            EvidenceDirectness.SPECULATIVE
        ),
        measurement_level=(
            MeasurementLevel.TEAM_MATCH
        ),
        current_source=(
            "Historical and World Cup match-result datasets."
        ),
        required_fields=(
            "event_id",
            "team_id",
            "goals_conceded",
        ),
        spatial_information_required=False,
        player_identity_required=False,
        sequence_context_required=False,
        notes=(
            "Too far downstream to determine which responsibility "
            "model generated the outcome."
        ),
    ),
)

HYPOTHESIS_IDS = (
    "double_pivot_shared_screen_v1",
    "double_pivot_side_specific_v1",
)


EVIDENCE_MAPPINGS = (
    HypothesisEvidenceMapping(
        hypothesis_id=(
            "double_pivot_shared_screen_v1"
        ),
        observable_id=(
            "dm_defensive_action_balance"
        ),
        predicted_tendency=(
            "The two DMs should display overlapping and relatively "
            "balanced defensive activity."
        ),
        discriminating_value=(
            "Moderate; balance is compatible with shared protection "
            "but does not prove it."
        ),
        validation_priority=(
            ValidationPriority.MEDIUM
        ),
        supports_shared_screen_if=(
            "Both DMs repeatedly show comparable defensive-action "
            "volumes across the sample."
        ),
        supports_side_specific_if=(
            "Activity is persistently specialized or highly "
            "asymmetric after controlling for minutes and quality."
        ),
        current_testability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        confounders=(
            "player quality",
            "injuries",
            "opponent strength",
            "role asymmetry",
            "minutes played",
        ),
    ),

    HypothesisEvidenceMapping(
        hypothesis_id=(
            "double_pivot_side_specific_v1"
        ),
        observable_id=(
            "dm_action_corridor_distribution"
        ),
        predicted_tendency=(
            "Each DM should concentrate defensive actions in the "
            "corridor matching the nearest center-back."
        ),
        discriminating_value=(
            "High; persistent corridor specialization directly "
            "distinguishes the side-specific candidate."
        ),
        validation_priority=(
            ValidationPriority.HIGH
        ),
        supports_shared_screen_if=(
            "Both DMs regularly act across both center-back "
            "corridors."
        ),
        supports_side_specific_if=(
            "DM1 is concentrated left and DM2 is concentrated right, "
            "or vice versa according to lineup orientation."
        ),
        current_testability=(
            EvidenceAvailability.REQUIRES_NEW_ENDPOINT
        ),
        confounders=(
            "formation changes",
            "opponent attacking side",
            "pressing traps",
            "player rotations",
            "substitutions",
        ),
    ),

    HypothesisEvidenceMapping(
        hypothesis_id=(
            "double_pivot_shared_screen_v1"
        ),
        observable_id=(
            "dm_to_cb_spatial_proximity"
        ),
        predicted_tendency=(
            "Each DM should maintain meaningful defensive proximity "
            "to both center-backs across sequences."
        ),
        discriminating_value=(
            "Very high; this is close to a direct observation of "
            "the proposed relationship."
        ),
        validation_priority=(
            ValidationPriority.DEFERRED
        ),
        supports_shared_screen_if=(
            "Both DMs repeatedly remain connected to both CBs."
        ),
        supports_side_specific_if=(
            "Each DM remains consistently closer to one CB."
        ),
        current_testability=(
            EvidenceAvailability.NOT_OBSERVABLE
        ),
        confounders=(
            "ball location",
            "opponent shape",
            "defensive block height",
            "transition state",
        ),
    ),

    HypothesisEvidenceMapping(
        hypothesis_id=(
            "double_pivot_side_specific_v1"
        ),
        observable_id=(
            "cb_step_out_cover_event"
        ),
        predicted_tendency=(
            "The same-side DM should usually cover behind a CB "
            "who steps out."
        ),
        discriminating_value=(
            "Very high; directly observes responsibility allocation."
        ),
        validation_priority=(
            ValidationPriority.DEFERRED
        ),
        supports_shared_screen_if=(
            "Either DM commonly covers either CB without stable "
            "lateral specialization."
        ),
        supports_side_specific_if=(
            "The same-side DM consistently covers the same-side CB."
        ),
        current_testability=(
            EvidenceAvailability.NOT_OBSERVABLE
        ),
        confounders=(
            "pressing scheme",
            "ball side",
            "rest defense",
            "temporary rotations",
        ),
    ),

    HypothesisEvidenceMapping(
        hypothesis_id=(
            "double_pivot_shared_screen_v1"
        ),
        observable_id=(
            "manual_sequence_annotation"
        ),
        predicted_tendency=(
            "Annotated sequences should show both DMs exchanging or "
            "sharing CB-protection responsibilities."
        ),
        discriminating_value=(
            "High if annotation agreement is strong."
        ),
        validation_priority=(
            ValidationPriority.HIGH
        ),
        supports_shared_screen_if=(
            "Both DMs are repeatedly coded as protecting both CBs."
        ),
        supports_side_specific_if=(
            "Each DM is repeatedly coded against one same-side CB."
        ),
        current_testability=(
            EvidenceAvailability.PARTIALLY_AVAILABLE
        ),
        confounders=(
            "annotator subjectivity",
            "camera angle",
            "sequence selection",
            "ambiguous tactical intent",
        ),
    ),

    HypothesisEvidenceMapping(
        hypothesis_id=(
            "double_pivot_side_specific_v1"
        ),
        observable_id=(
            "goals_conceded"
        ),
        predicted_tendency=(
            "No reliable directional prediction can be assigned "
            "without many additional assumptions."
        ),
        discriminating_value=(
            "Very low; goals conceded cannot identify responsibility "
            "allocation."
        ),
        validation_priority=(
            ValidationPriority.LOW
        ),
        supports_shared_screen_if=(
            "No defensible direct criterion."
        ),
        supports_side_specific_if=(
            "No defensible direct criterion."
        ),
        current_testability=(
            EvidenceAvailability.CURRENTLY_AVAILABLE
        ),
        confounders=(
            "player quality",
            "goalkeeping",
            "opponent quality",
            "finishing variance",
            "set pieces",
            "game state",
        ),
    ),
)

def build_observable_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "observable_id":
                    observable.observable_id,
                "observable_name":
                    observable.name,
                "description":
                    observable.description,
                "availability":
                    observable.availability.value,
                "directness":
                    observable.directness.value,
                "measurement_level":
                    observable.measurement_level.value,
                "current_source":
                    observable.current_source,
                "required_fields":
                    "|".join(
                        observable.required_fields
                    ),
                "spatial_information_required":
                    observable
                    .spatial_information_required,
                "player_identity_required":
                    observable
                    .player_identity_required,
                "sequence_context_required":
                    observable
                    .sequence_context_required,
                "notes":
                    observable.notes,
            }
            for observable in OBSERVABLES
        ]
    )


def build_evidence_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis_id":
                    mapping.hypothesis_id,
                "observable_id":
                    mapping.observable_id,
                "predicted_tendency":
                    mapping.predicted_tendency,
                "discriminating_value":
                    mapping.discriminating_value,
                "validation_priority":
                    mapping
                    .validation_priority.value,
                "supports_shared_screen_if":
                    mapping
                    .supports_shared_screen_if,
                "supports_side_specific_if":
                    mapping
                    .supports_side_specific_if,
                "current_testability":
                    mapping
                    .current_testability.value,
                "confounders":
                    "|".join(
                        mapping.confounders
                    ),
            }
            for mapping
            in EVIDENCE_MAPPINGS
        ]
    )

def build_observable_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "observable_id":
                    observable.observable_id,
                "observable_name":
                    observable.name,
                "description":
                    observable.description,
                "availability":
                    observable.availability.value,
                "directness":
                    observable.directness.value,
                "measurement_level":
                    observable.measurement_level.value,
                "current_source":
                    observable.current_source,
                "required_fields":
                    "|".join(
                        observable.required_fields
                    ),
                "spatial_information_required":
                    observable
                    .spatial_information_required,
                "player_identity_required":
                    observable
                    .player_identity_required,
                "sequence_context_required":
                    observable
                    .sequence_context_required,
                "notes":
                    observable.notes,
            }
            for observable in OBSERVABLES
        ]
    )


def build_evidence_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis_id":
                    mapping.hypothesis_id,
                "observable_id":
                    mapping.observable_id,
                "predicted_tendency":
                    mapping.predicted_tendency,
                "discriminating_value":
                    mapping.discriminating_value,
                "validation_priority":
                    mapping
                    .validation_priority.value,
                "supports_shared_screen_if":
                    mapping
                    .supports_shared_screen_if,
                "supports_side_specific_if":
                    mapping
                    .supports_side_specific_if,
                "current_testability":
                    mapping
                    .current_testability.value,
                "confounders":
                    "|".join(
                        mapping.confounders
                    ),
            }
            for mapping
            in EVIDENCE_MAPPINGS
        ]
    )

def build_data_capability_audit() -> pd.DataFrame:
    rows = (
        {
            "capability":
                "full_match_team_statistics",
            "currently_available": True,
            "granularity": "team_match",
            "player_identity": False,
            "spatial_coordinates": False,
            "sequence_context": False,
            "validation_use":
                "context and outcome controls only",
        },
        {
            "capability":
                "player_per90_defensive_statistics",
            "currently_available": True,
            "granularity": "player_season",
            "player_identity": True,
            "spatial_coordinates": False,
            "sequence_context": False,
            "validation_use":
                "weak or moderate responsibility proxies",
        },
        {
            "capability":
                "formation_slot_assignments",
            "currently_available": True,
            "granularity": "expected_lineup",
            "player_identity": True,
            "spatial_coordinates": False,
            "sequence_context": False,
            "validation_use":
                "identify candidate DMs and CBs",
        },
        {
            "capability":
                "abstract_formation_geometry",
            "currently_available": True,
            "granularity": "formation_template",
            "player_identity": False,
            "spatial_coordinates": True,
            "sequence_context": False,
            "validation_use":
                "define theoretical corridors only",
        },
        {
            "capability":
                "player_event_locations",
            "currently_available": False,
            "granularity": "spatial_event",
            "player_identity": True,
            "spatial_coordinates": True,
            "sequence_context": False,
            "validation_use":
                "required for corridor specialization",
        },
        {
            "capability":
                "synchronized_player_tracking",
            "currently_available": False,
            "granularity": "defensive_sequence",
            "player_identity": True,
            "spatial_coordinates": True,
            "sequence_context": True,
            "validation_use":
                "direct DM-to-CB responsibility testing",
        },
        {
            "capability":
                "manual_video_annotation",
            "currently_available": False,
            "granularity": "defensive_sequence",
            "player_identity": True,
            "spatial_coordinates": True,
            "sequence_context": True,
            "validation_use":
                "feasible direct evidence with human labor",
        },
    )

    return pd.DataFrame(rows)

def validate_outputs(
    *,
    observable_catalog: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
    capability_audit: pd.DataFrame,
) -> None:
    if (
        observable_catalog.empty
        or evidence_matrix.empty
        or capability_audit.empty
    ):
        raise AssertionError(
            "At least one Study 106A output is empty."
        )

    if observable_catalog[
        "observable_id"
    ].duplicated().any():
        raise AssertionError(
            "Observable catalog contains duplicate IDs."
        )

    known_observables = set(
        observable_catalog[
            "observable_id"
        ].astype(str)
    )

    mapped_observables = set(
        evidence_matrix[
            "observable_id"
        ].astype(str)
    )

    if not mapped_observables.issubset(
        known_observables
    ):
        raise AssertionError(
            "Evidence matrix references unknown observables."
        )

    if not set(
        evidence_matrix[
            "hypothesis_id"
        ].astype(str)
    ).issubset(
        set(HYPOTHESIS_IDS)
    ):
        raise AssertionError(
            "Evidence matrix references unknown hypotheses."
        )

    direct_current = observable_catalog.loc[
        observable_catalog[
            "directness"
        ].eq(
            EvidenceDirectness.DIRECT.value
        )
        & observable_catalog[
            "availability"
        ].eq(
            EvidenceAvailability
            .CURRENTLY_AVAILABLE
            .value
        )
    ]

    if not direct_current.empty:
        raise AssertionError(
            "The current pipeline was incorrectly claimed to "
            "contain directly observed responsibility evidence."
        )

    spatial_current = observable_catalog.loc[
        observable_catalog[
            "spatial_information_required"
        ]
        & observable_catalog[
            "availability"
        ].eq(
            EvidenceAvailability
            .CURRENTLY_AVAILABLE
            .value
        )
    ]

    if not spatial_current.empty:
        raise AssertionError(
            "A spatial observable was incorrectly marked as "
            "currently available."
        )

    if not evidence_matrix[
        "current_testability"
    ].isin(
        {
            availability.value
            for availability
            in EvidenceAvailability
        }
    ).all():
        raise AssertionError(
            "Evidence matrix contains an invalid availability state."
        )

def write_protocol(
    *,
    observable_catalog: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
    capability_audit: pd.DataFrame,
) -> None:
    availability_summary = (
        observable_catalog
        .groupby(
            [
                "availability",
                "directness",
            ],
            as_index=False,
        )
        .agg(
            observable_count=(
                "observable_id",
                "count",
            )
        )
    )

    priority_summary = (
        evidence_matrix
        .groupby(
            [
                "validation_priority",
                "current_testability",
            ],
            as_index=False,
        )
        .agg(
            mapping_count=(
                "observable_id",
                "count",
            )
        )
    )

    protocol = f"""# Study 106A — Football Evidence Mapping Protocol

## Status

**PASS**

## Purpose

Define the methodological bridge between structural football
hypotheses and observable match evidence.

## Current evidence boundary

The current pipeline contains:

- expected lineups and tactical roles;
- abstract formation geometry;
- player-level per-90 defensive statistics;
- team-level full-match statistics;
- match outcomes and competition metadata.

The current pipeline does not contain:

- player-event coordinates;
- synchronized tracking data;
- defensive-sequence labels;
- observed DM-to-CB responsibility assignments;
- direct measures of communication or tactical intent.

## Observable availability summary

{availability_summary.to_markdown(index=False)}

## Evidence-mapping priority summary

{priority_summary.to_markdown(index=False)}

## Current data-capability audit

{capability_audit.to_markdown(index=False)}

## Core methodological rule

Observation and interpretation must remain separate.

For example:

- Observation: a DM recorded 6.2 recoveries per 90.
- Interpretation: this may be consistent with shared-screen activity.

The observation must not be stored as proof of the interpretation.

## Double-pivot validation conclusion

The current pipeline can support an exploratory comparison of
double-pivot defensive workloads.

It cannot directly distinguish shared-screen responsibility from
side-specific responsibility because it lacks player-level spatial
and sequence data.

## Recommended empirical pathway

1. Build a player-season double-pivot workload comparison using the
   existing per-90 data.
2. Treat the result as proxy evidence only.
3. Investigate whether a reliable player-event-location endpoint can
   be collected.
4. If spatial events are unavailable, design a small manual video
   annotation pilot.
5. Do not promote either candidate from aggregate statistics alone.

## Interpretation boundary

This study maps possible evidence. It does not collect new evidence,
validate either hypothesis, or change any football model.
"""

    PROTOCOL_PATH.write_text(
        protocol,
        encoding="utf-8",
    )

def write_backlog() -> None:
    backlog = """# Football Research Backlog

## double_pivot_shared_screen_v1

**Status:** empirical validation required

**Current evidence:** weak and moderate aggregate proxies only

**Highest-value missing evidence:**
player-level defensive action locations and DM-to-CB proximity

**Next feasible study:**
player-season double-pivot workload pilot

**Promotion eligibility:** no

---

## double_pivot_side_specific_v1

**Status:** empirical validation required

**Current evidence:** no direct corridor evidence

**Highest-value missing evidence:**
left/right defensive-action distributions for each DM

**Next feasible study:**
spatial-event endpoint feasibility audit

**Promotion eligibility:** no

---

## dm_protects_cb_v1

**Status:** active diagnostic for single-pivot 4-3-3

**Current evidence:** expert structural hypothesis only

**Next evidence:**
single-pivot defensive-sequence validation

**Production eligibility:** no

---

## dm_supports_cm_v1

**Status:** revision required

**Reason:**
the meaning of support is under-specified and no direct observable has
yet been registered

**Next action:**
decompose support into narrower observable claims

---

## cm_supports_w_v1

**Status:** revision required

**Reason:**
support could mean passing availability, defensive cover, overlap, or
progression support

**Next action:**
separate these concepts before empirical mapping

---

## dm_connects_cb_cm_v1

**Status:** deferred

**Reason:**
connection is structurally plausible but currently lacks a precise
observable definition

**Next action:**
define whether connection is positional, passing-based, or
sequence-based
"""

    BACKLOG_PATH.write_text(
        backlog,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 106A — FOOTBALL EVIDENCE "
        "MAPPING FOUNDATIONS"
    )
    print("=" * 88)

    observable_catalog = (
        build_observable_catalog()
    )

    evidence_matrix = (
        build_evidence_matrix()
    )

    capability_audit = (
        build_data_capability_audit()
    )

    validate_outputs(
        observable_catalog=observable_catalog,
        evidence_matrix=evidence_matrix,
        capability_audit=capability_audit,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    observable_catalog.to_csv(
        OBSERVABLE_CATALOG_PATH,
        index=False,
    )

    evidence_matrix.to_csv(
        EVIDENCE_MATRIX_PATH,
        index=False,
    )

    capability_audit.to_csv(
        DATA_CAPABILITY_PATH,
        index=False,
    )

    write_protocol(
        observable_catalog=observable_catalog,
        evidence_matrix=evidence_matrix,
        capability_audit=capability_audit,
    )

    write_backlog()

    availability_counts = (
        observable_catalog[
            "availability"
        ]
        .value_counts()
        .to_dict()
    )

    metadata = {
        "study_id": "106A",
        "study_name": (
            "Football Evidence Mapping Foundations"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "observable_count": len(
            observable_catalog
        ),
        "evidence_mapping_count": len(
            evidence_matrix
        ),
        "hypothesis_count": int(
            evidence_matrix[
                "hypothesis_id"
            ].nunique()
        ),
        "currently_available_observables": int(
            availability_counts.get(
                EvidenceAvailability
                .CURRENTLY_AVAILABLE
                .value,
                0,
            )
        ),
        "partially_available_observables": int(
            availability_counts.get(
                EvidenceAvailability
                .PARTIALLY_AVAILABLE
                .value,
                0,
            )
        ),
        "new_endpoint_observables": int(
            availability_counts.get(
                EvidenceAvailability
                .REQUIRES_NEW_ENDPOINT
                .value,
                0,
            )
        ),
        "not_observable_count": int(
            availability_counts.get(
                EvidenceAvailability
                .NOT_OBSERVABLE
                .value,
                0,
            )
        ),
        "direct_responsibility_evidence_currently_available":
            False,
        "empirical_validation_performed":
            False,
        "hypothesis_promoted": False,
        "canonical_hypothesis_register_changed":
            False,
        "football_graph_changed": False,
        "weights_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "This study maps hypotheses to possible observable "
            "evidence. It does not validate either double-pivot "
            "candidate or establish causal football relationships."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Evidence mapping summary")
    print("-" * 88)
    print(
        "  Observables registered: "
        f"{metadata['observable_count']}"
    )
    print(
        "  Hypothesis-evidence mappings: "
        f"{metadata['evidence_mapping_count']}"
    )
    print(
        "  Hypotheses represented: "
        f"{metadata['hypothesis_count']}"
    )
    print(
        "  Currently available observables: "
        f"{metadata['currently_available_observables']}"
    )
    print(
        "  Partially available observables: "
        f"{metadata['partially_available_observables']}"
    )
    print(
        "  Observables requiring new endpoint: "
        f"{metadata['new_endpoint_observables']}"
    )
    print(
        "  Currently unavailable observables: "
        f"{metadata['not_observable_count']}"
    )
    print(
        "  Direct responsibility evidence available: NO"
    )
    print(
        "  Empirical validation performed: NO"
    )
    print(
        "  Hypothesis promoted: NO"
    )
    print(
        "  Team strength changed: NO"
    )
    print(
        "  Simulation run: NO"
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()