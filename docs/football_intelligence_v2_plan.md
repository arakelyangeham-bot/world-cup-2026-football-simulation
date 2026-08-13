football_intelligence_v2_plan.md

# Football Intelligence v2 Plan

## Purpose

Football Intelligence v2 is the next development phase of the World Cup 2026 prediction platform.

Its goal is to improve the quality of football information entering the existing prediction and simulation system without redesigning the production simulator.

## Core Principle

The production engine should remain stable.

Football Intelligence v2 should improve this layer:

```text
Player Features
  ↓
Aggregation
  ↓
Team Repository
```

It should not initially rewrite:

```text
MatchPredictor
MatchSampler
Match Engine
Tournament Simulator
Monte Carlo Driver
Outcome Evaluation
```

## Current v1 Pipeline

The current football-intelligence pipeline is:

```text
Roster + player stats
  ↓
Merged player dataset
  ↓
Per-90 player features
  ↓
Position-group aggregation
  ↓
Team composites
  ↓
Team Repository
```

This already gives the platform a working player-to-team abstraction.

## v2 Objective

The v2 objective is to move from static team strength toward dynamic, context-aware team strength.

```text
Static team ratings
  ↓
Dynamic team repository
```

## Proposed v2 Layers

```text
Player Data
  ↓
Player Feature Store
  ↓
Availability Layer
  ↓
Expected Lineup Layer
  ↓
Lineup Aggregation Layer
  ↓
Dynamic Team Repository
  ↓
Existing Prediction Layer
```

## Layer Responsibilities

### 1. Player Feature Store

Stores normalized player attributes.

Examples:

```text
attacking contribution
creative contribution
defensive contribution
goalkeeping contribution
possession contribution
minutes played
position group
recent form
```

### 2. Availability Layer

Tracks whether players are usable for a match.

Examples:

```text
available
injured
suspended
doubtful
not selected
fitness-limited
```

### 3. Expected Lineup Layer

Produces expected starters, bench players, and expected minutes.

Examples:

```text
starting XI
substitutes
expected minutes
formation
position assignment
```

### 4. Lineup Aggregation Layer

Aggregates player-level information into team-level values.

Examples:

```text
attack
midfield
defense
gk
poisson_attack
poisson_defense
depth
volatility
availability_penalty
```

### 5. Dynamic Team Repository

Exports the same canonical interface expected by the production prediction layer.

The first goal should be compatibility, not sophistication.

## Design Requirement

The existing prediction layer should continue to receive team dictionaries.

That means Football Intelligence v2 should initially preserve fields such as:

```text
attack
midfield
defense
gk
poisson_attack
poisson_defense
fifa_points
```

Additional fields may be added later, but the original fields should remain stable.

## First Milestone: FI-2.0 Skeleton

The first implementation milestone should create structure, not model changes.

Suggested files:

```text
src/football_intelligence/
  player_schema.py
  team_schema.py
  availability_schema.py
  lineup_schema.py
  aggregation_schema.py
  dynamic_team_repository.py
```

No production behavior should change in this milestone.

## Second Milestone: FI-2.1 Repository Parity

Build a dynamic repository generator that reproduces the current static repository values.

Goal:

```text
new football_intelligence pipeline output
  ≈
current load_team_repository() output
```

This validates that the new layer can replace the old pipeline without changing behavior.

## Third Milestone: FI-2.2 Expected Lineup Prototype

Introduce expected lineup logic using simple deterministic assumptions.

Examples:

```text
top minutes by position group
top rating by position group
minimum positional coverage
bench depth calculation
```

This should be benchmarked against the existing static team repository before promotion.

## Fourth Milestone: FI-2.3 Availability Prototype

Add availability inputs without changing model architecture.

Examples:

```text
remove unavailable players
reduce doubtful players' expected minutes
apply squad-selection constraints
recompute team strengths
```

## Benchmarking Requirements

Every meaningful change must be benchmarked against the current production baseline.

Required metrics:

```text
multiclass Brier score
multiclass log loss
calibration tables
expected calibration error
scoreline realism TVD
tournament probability stability
```

No football-intelligence enhancement should be promoted unless it improves or preserves the production benchmark profile.

## Promotion Rule

A Football Intelligence v2 change may be promoted only if:

```text
1. It preserves production interfaces.
2. It passes repository validation.
3. It improves or maintains historical prediction metrics.
4. It does not degrade scoreline realism.
5. It does not destabilize tournament outputs without explanation.
```

## Non-Goals for Early v2

Do not start with:

```text
deep learning player models
complex chemistry models
manual tactical modeling
formation-specific neural models
live injury scraping
real-time transfer integration
```

These may become future research topics, but they should not define the first v2 milestone.

## Architectural Philosophy

Football Intelligence v2 should be incremental.

The goal is not to make the system complicated.

The goal is to make the existing Team Repository smarter, more explainable, and more context-aware while preserving the mature production simulation platform.

