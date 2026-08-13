# World Cup 2026 Platform Architecture v1.0

## Purpose

This document records the current production architecture before beginning the Football Intelligence Phase.

The platform is now stable enough that future work should improve football information flowing into the system, rather than redesigning the prediction or simulation engine.

## Core Architecture

```text
Raw Data
  ↓
Player / Team Feature Construction
  ↓
Team Strength Aggregation
  ↓
Canonical Team Repository
  ↓
Prediction Layer
  ↓
Match Engine
  ↓
Competition Engine
  ↓
Monte Carlo Tournament Simulation
```

## Stable Architectural Seam

The most important interface is the **Team Repository**.

It exposes canonical team-level attributes:

```text
attack
midfield
defense
gk
poisson_attack
poisson_defense
fifa_points
```

This repository is the boundary between football knowledge and prediction mechanics.

## Existing Football Intelligence v1

The current football-intelligence layer already exists in first-generation form:

```text
Roster + Sofascore player stats
  ↓
Merged player dataset
  ↓
Per-90 player features
  ↓
Position-group aggregation
  ↓
Team composites
  ↓
Poisson attack / defense adjustments
```

This pipeline is implemented through the roster merge, feature engineering, and team aggregation scripts.

## Prediction Layer

The prediction layer is intentionally model-facing rather than football-facing.

The `MatchPredictor` accepts home and away team dictionaries, builds engineered features, validates them, and passes them to the production model.

Expected goals are handled separately through the lambda model interface, which supports both heuristic and calibrated models.

## Simulation Layer

The match engine converts expected goals into scorelines using the configured goal sampler. It does not contain player, lineup, or roster logic.

The match sampler converts predicted probabilities into sampled outcomes and handles knockout winner resolution when draws are sampled.

## Evaluation Layer

The evaluation framework is predictor-agnostic. It evaluates any predictor that exposes the expected prediction interface, producing accuracy, log loss, confusion matrices, and prediction outputs.

This is important because future football-intelligence improvements can be tested without rewriting evaluation infrastructure.

## Architectural Conclusion

The platform should not be redesigned.

The stable lower layers are:

```text
Prediction
Match Engine
Competition Engine
Monte Carlo Simulation
Evaluation
```

The layer that should evolve is:

```text
Player Features
  ↓
Aggregation
  ↓
Team Repository
```

## Football Intelligence Phase Direction

Football Intelligence v2 should improve the way team repository values are produced.

Future enhancements may include:

```text
expected lineups
injuries
suspensions
availability
bench depth
player minutes weighting
position-specific contribution models
recency weighting
dynamic team strength
```

These enhancements should feed the existing Team Repository interface whenever possible.

## Non-Goals

Football Intelligence v2 should not initially rewrite:

```text
MatchPredictor
MatchSampler
Match Engine
Tournament Simulator
Monte Carlo Driver
Outcome Evaluation
```

Those layers are already well-separated and production-stable.

## Guiding Principle

Future work should make the Team Repository smarter while preserving the existing prediction and simulation interfaces.

The goal is not a new simulator.

The goal is better football knowledge entering the simulator.
