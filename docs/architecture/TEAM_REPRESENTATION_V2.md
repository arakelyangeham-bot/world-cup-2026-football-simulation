TEAM_REPRESENTATION_V2

# Team Representation Version 2

## Status

Implementation specification

## Purpose

A Team Representation is a compact numerical description of a football team's
player-derived ability under a defined player-selection context.

It answers:

> How strong is this team, given the players included in this representation?

It does not describe:

- the competition environment;
- the venue;
- home advantage;
- the opponent;
- tactical matchups;
- expected goals;
- scoreline probabilities.

Those belong to later prediction layers.

---

# Construction model

A Team Representation is produced by:

```text
Player representations
        +
Player-selection context
        +
Aggregation strategy
        ↓
Team Representation

The same team may therefore have several valid representations.

Examples:

France — full squad
France — available squad
France — expected starting XI
France — historical starting XI

These representations are not interchangeable and must record how they were
constructed.

Version 2 fields
Identity
Field	Description
team	Canonical team name
team_type	national_team or club
Construction context
Field	Description
representation_type	Player-selection context used
aggregation_profile	Name of the aggregation strategy profile
player_count	Number of players used
available_player_count	Number of included players marked available

Recommended representation_type values:

full_squad
available_squad
expected_starting_xi
historical_starting_xi
manual
Core dimensions
Field	Description
attack	Aggregated attacking strength
midfield	Aggregated midfield strength
defense	Aggregated defensive strength
goalkeeper	Aggregated goalkeeper strength
Depth dimensions
Field	Description
attack_depth	Attacking quality across the selected player population
midfield_depth	Midfield quality across the selected player population
defense_depth	Defensive quality across the selected player population
General quality and reliability
Field	Description
squad_quality	Mean overall quality of included players
evidence_score	Proportion of included players with usable rating evidence
Representation versus repository

The Team Representation is the analytical object.

The team repository is a serialized feature store used by downstream systems.

Team Representation
        ↓
Repository projection
        ↓
CSV or dictionary

Repository projection may add external priors such as:

rating_prior
fifa_points

These are not player-derived properties and therefore do not belong inside the
core Team Representation.

Similarly, Poisson attack and defense values are model-facing projections.
They may initially equal attack and defense, but they should remain repository
or model-adapter fields rather than fundamental representation fields.

Representation versus match context

Competition environment and venue information must remain separate.

Home Team Representation
+
Away Team Representation
+
Match Context
        ↓
Expected Goals Model

Match Context may eventually contain:

competition_key
season
environment_pc1
environment_pc2
home_advantage
venue
neutral_site

This prevents the same team representation from changing merely because the
team enters a different competition.

Aggregation profiles

The project currently contains two forms of aggregation logic.

One representation builder uses:

attack     = top-five mean
midfield   = top-five mean
defense    = top-five mean
goalkeeper = best player

A separate dimension-aggregation module supports named strategies such as:

star_weighted
starter_plus_depth
top_11_mean
top_5_mean
best_player

Version 2 must record the selected aggregation profile, but it does not yet
require these implementations to be unified.

Recommended initial profile names:

legacy_top_5
dimension_specific_default

Where:

legacy_top_5
    attack     -> top_5_mean
    midfield   -> top_5_mean
    defense    -> top_5_mean
    goalkeeper -> best_player

and:

dimension_specific_default
    attack     -> star_weighted
    midfield   -> starter_plus_depth
    defense    -> top_11_mean
    goalkeeper -> best_player
Version 2 dataclass

The target representation is:

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamRepresentation:
    team: str
    team_type: str
    representation_type: str
    aggregation_profile: str

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    attack_depth: float
    midfield_depth: float
    defense_depth: float

    squad_quality: float
    evidence_score: float

    player_count: int
    available_player_count: int
Compatibility policy

Version 2 must initially preserve the existing national_team interface.

The first implementation may therefore keep:

national_team: str

instead of immediately renaming it to:

team: str

Renaming should occur only after all usages have been located and tested.

The safe first change is to add metadata fields while preserving all existing
fields and return values.

First implementation phase

Add these fields to the existing dataclass:

representation_type
aggregation_profile
player_count
available_player_count

Use defaults that preserve current behavior:

representation_type = "full_squad"
aggregation_profile = "legacy_top_5"

The builder should calculate:

player_count = number of included players
available_player_count = number marked available

No strength formula should change during this phase.

Second implementation phase

Propagate the new fields through:

TeamRepresentation
        ↓
TeamRepositoryEntry
        ↓
Player Intelligence Repository CSV
        ↓
Canonical repository loader

All existing public repository keys must remain unchanged.

The new fields should initially be additive and optional for older repository
files.

Non-goals

Version 2 does not yet:

unify all aggregation implementations;
change current strength values;
introduce club-specific logic;
add environmental PCA coordinates;
create a match snapshot object;
alter the expected-goals model;
alter the scoreline engine;
alter tournament simulation.
Acceptance criteria

Version 2 is complete when:

Existing team strength values are numerically unchanged.
Every new representation records its player-selection context.
Every new representation records its aggregation profile.
Player counts are preserved through repository serialization.
Older repository CSV files still load successfully.
Existing simulator benchmarks produce identical results.