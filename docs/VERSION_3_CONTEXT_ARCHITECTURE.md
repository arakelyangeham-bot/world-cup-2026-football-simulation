# Version 3 Context Architecture

## Status

**FOUNDATIONAL DESIGN — DRAFT 1**

## Purpose

Version 3 extends the football-intelligence layer by preserving and using football context that was previously discarded before team aggregation.

The central architectural principle is:

> Context should be represented before player information is projected and aggregated.

Version 2 established a stable downstream contract:

```text
Player-derived information
    -> TeamRepresentation
    -> Repository
    -> Observation
    -> Goal model
    -> Prediction pipeline
```

Version 3 must enrich the information flowing into `TeamRepresentation` without changing downstream production interfaces unless evidence later requires it.

## Current information flow

```text
Formation manifest
    -> StartingXIBuilder
    -> StartingXI
    -> build_team_representation_from_starting_xi()
    -> build_team_representation_from_players()
    -> role projections
    -> scalar aggregation
    -> TeamRepresentation
```

The formation manifest supplies tactical slots and required roles. `StartingXIBuilder` uses those rows to select the highest-rated eligible player for each role while preventing duplicate selections.

Study 096A introduced `LineupAssignment`, allowing `StartingXI` to retain:

- slot;
- assigned tactical role;
- selected player;
- role-specific selection rating.

The historical `StartingXI.players` tuple remains available for backward compatibility.

## Current information bottleneck

The team-representation builder still consumes only:

```python
starting_xi.players
```

The preserved assignments are not yet used.

Each player's role-rating matrix is converted into four context-free scalar projections:

- attack;
- midfield;
- defense;
- goalkeeper.

The aggregation layer then receives anonymous scalar lists. At that point, the system cannot inspect:

- formation slot;
- assigned tactical role;
- left/right or central deployment;
- whether the assigned role is the player's strongest role;
- neighboring players;
- tactical-unit structure;
- formation balance;
- selection compromise.

This is the principal Version 3 modeling seam.

## Foundational design decision

Version 3 should not make the scalar aggregation adapter reconstruct football context.

Instead, context should be represented explicitly before projection:

```text
Player
    -> LineupAssignment
    -> FootballContext
    -> context-aware player projection
    -> aggregation
    -> TeamRepresentation
```

This keeps aggregation mathematically focused while allowing projection to account for actual deployment.

## Core domain objects

### LineupAssignment

Current fields:

```text
slot
tactical_role
player
selection_rating
```

Its initial responsibility is information preservation, not strength adjustment.

### StartingXI

Current responsibilities:

```text
national_team
formation
players
assignments
```

Version 2 consumers may continue using `players`. Version 3 consumers should prefer `assignments` when tactical context matters.

### FootballContext

Proposed initial contract:

```python
@dataclass(frozen=True)
class FootballContext:
    national_team: str
    formation: str
    assignments: tuple[LineupAssignment, ...]
```

Possible derived properties:

```text
role occupancy
role-fit distribution
unit membership
left/right balance
central-lane balance
natural-role coverage
formation completeness
```

Derived values should not be persisted when they can be reproduced from assignments.

## Role vocabulary

The current role vocabulary is:

```text
GK
CB
FB
DM
CM
AM
WM
W
ST
```

It is shared by player role ratings, formation requirements, eligibility, selection, and role projection.

Version 3 should not immediately replace this vocabulary. The first studies should test how much context can be extracted from:

```text
formation
slot
tactical_role
player role ratings
```

Only evidence should justify a larger tactical ontology.

## Context-aware projection

The current projection functions are global:

```text
project_attack(role_ratings)
project_midfield(role_ratings)
project_defense(role_ratings)
project_goalkeeper(role_ratings)
```

The future seam should support something conceptually similar to:

```python
project_assignment(
    assignment,
    football_context,
)
```

The output may remain:

```text
attack
midfield
defense
goalkeeper
```

This preserves the downstream `TeamRepresentation` contract.

The first context-aware projection must be deterministic, interpretable, and benchmarkable.

## Role-fit concept

The first contextual signal should be diagnostic role fit:

```text
role_fit_ratio =
    assigned_role_rating / best_available_role_rating
```

Possible interpretation:

```text
1.00     strongest rated role
0.90     mild compromise
0.75     substantial compromise
```

No penalty or bonus should be applied until the empirical distribution and predictive value are studied.

Safeguards:

- missing assigned-role ratings must be explicit;
- non-positive denominators require deliberate handling;
- eligibility and rating availability must not be conflated;
- role fit must not silently alter selection;
- raw and transformed values must remain auditable.

## Tactical units

Future studies may derive:

```text
goalkeeper
back line
midfield line
front line
left channel
central channel
right channel
center-back pair
fullback-winger pairs
pivot group
creator-finisher group
```

Unit definitions must be explicit and formation-specific.

## Compatibility policy

Version 3 additions remain additive until benchmark evidence justifies promotion.

The following interfaces should remain unchanged during the architecture phase:

```text
TeamRepresentation
ProductionClubRepository
LiveMatchObservationBuilder
ProductionGoalModel
ProductionPredictionPipeline
match engine
tournament simulator
```

Legacy behavior remains the control.

## Research sequence

### Study 096A — Assignment preservation

Status: **PASS**

Established:

- slot preservation;
- tactical-role preservation;
- selection-rating preservation;
- player-order preservation;
- backward-compatible `StartingXI.players`;
- zero aggregation drift;
- full player-intelligence regression success.

### Study 096B — Real-lineup context audit

Purpose:

- describe assignment-level role fit;
- validate slot and role vocabularies;
- measure compromised assignments;
- identify real formation-context patterns;
- make no team-strength changes.

Expected outputs:

```text
lineup_assignment_population.csv
team_context_summary.csv
role_fit_distribution.csv
slot_role_vocabulary_audit.csv
study_096b_metadata.json
STUDY_096B_REPORT.md
```

### Study 097 — Context-aware projection prototype

Purpose:

- introduce an additive projection strategy;
- preserve the team-representation schema;
- compare context-free and context-aware projections;
- avoid production promotion.

### Study 098 — Synthetic formation benchmark

Purpose:

- construct lineups with equal player quality but different role fit and structural balance;
- verify that the new representation distinguishes situations the Version 2 representation treats as equivalent.

### Study 099 — Real-football predictive benchmark

Purpose:

- compare legacy and context-aware representations on matched historical fixtures;
- reuse frozen splits, targets, ClubElo priors, and metrics;
- determine whether context adds predictive signal.

## Non-goals

Version 3 should not initially attempt to model:

- manager tactics;
- learned chemistry embeddings;
- neural tactical representations;
- live substitutions;
- fatigue dynamics;
- opponent-specific plans;
- automatic formation inference;
- a complete football ontology.

## Decision rules

A contextual feature may enter production only after:

```text
schema validation
backward compatibility
synthetic behavioral validation
real-data stability
matched predictive benchmark
robustness analysis
production-path replay
```

## Architectural principle

Version 2 solved modularity.

Version 3 should solve context.

> Do not ask aggregation to infer football structure from anonymous numbers. Preserve the structure explicitly, represent it before projection, and keep the downstream production contract stable.
