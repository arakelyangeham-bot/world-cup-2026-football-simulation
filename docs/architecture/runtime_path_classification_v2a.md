runtime_path_classification_v2a

# Runtime Path Classification — Version 2A

## Purpose

This document classifies every significant runtime pathway in the
Version 2A architecture.

The objective is to remove ambiguity regarding:

- the official production interfaces;
- active runtime implementations;
- compatibility layers;
- research-only components;
- historical infrastructure.

This document is descriptive rather than prescriptive.

No component should be removed solely because it is classified as
legacy.

---

# Classification Levels

## AUTHORITATIVE

Primary public interface.

New production development should target these interfaces.

---

## ACTIVE

Currently used in production or simulation.

Supported and maintained.

---

## DOMAIN-SPECIFIC

Active runtime intended for a particular football domain.

Examples include:

- club football;
- national-team football.

---

## COMPATIBILITY

Maintained to preserve older workflows or provide lower-level
access.

Generally not recommended as the first public entry point.

---

## RESEARCH

Supports experimentation, benchmarking, validation or scientific
studies.

Not intended as production runtime.

---

## LEGACY

Historical implementation retained for reproducibility.

Not recommended for future architectural expansion.

---

# Prediction Runtime

## ProductionPredictionPipeline

Classification

```
AUTHORITATIVE
```

Purpose

Deterministic production prediction API.

Responsibilities

- build observations;
- predict expected goals;
- calculate outcome probabilities;
- validate runtime contracts;
- expose immutable prediction objects.

Recommended public interface:

```python
predict_fixture(...)
predict_fixtures(...)
```

Future runtime work should begin here.

---

## ProductionPredictionPipelineFactory

Classification

```
ACTIVE
```

Purpose

Assemble the production runtime from persisted artifacts.

Responsibilities

- repository loading;
- ClubElo loading;
- observation builder construction;
- goal-model loading;
- pipeline construction.

---

## LiveMatchObservationBuilder

Classification

```
ACTIVE
```

Purpose

Construct deterministic live football observations.

Responsibilities

- repository lookup;
- ClubElo resolution;
- feature construction;
- contract validation.

---

## ProductionGoalModel

Classification

```
ACTIVE
```

Purpose

Runtime wrapper around frozen expected-goal artifacts.

Responsibilities

- artifact loading;
- feature validation;
- expected-goal prediction.

Performs no fitting.

---

## ScorelineProbabilityCalculator

Classification

```
ACTIVE
```

Purpose

Pure mathematical conversion:

```
Expected goals

↓

Outcome probabilities
```

Contains no football-specific logic.

---

## IntegratedClubGoalPredictor

Classification

```
COMPATIBILITY
```

Purpose

Lower-level wrapper around ProductionGoalModel.

Provides:

- artifact defaults;
- prediction-date validation;
- provenance.

Future external code should generally prefer:

```
ProductionPredictionPipeline
```

---

# Production Artifacts

## ProductionClubRepositoryBuilder

Classification

```
AUTHORITATIVE
```

Purpose

Build immutable production repositories.

Responsibilities

- serialization;
- validation;
- persistence.

Does not calculate football intelligence.

---

## ProductionClubRepository

Classification

```
ACTIVE
```

Purpose

Read-only runtime repository.

Provides deterministic access to persisted club
representations.

---

## ProductionRepositorySchema

Classification

```
AUTHORITATIVE
```

Defines:

```
ProductionClubRecord
```

Represents persisted runtime contracts.

---

# Rating Priors

## ClubEloRepository

Classification

```
ACTIVE
```

Purpose

Prediction-date club-strength priors.

Responsibilities

- cache loading;
- temporal resolution;
- validity checking.

Competition-agnostic.

---

# Club Simulation

## FootballModelAdapter

Classification

```
AUTHORITATIVE
```

Purpose

Bridge between production prediction and competition
simulation.

Stable interface:

```python
simulate_match(...)
```

Competition code should depend on this abstraction rather than
specific football models.

---

## LeagueMatchSimulator

Classification

```
ACTIVE
```

Purpose

Convert scheduled fixtures into MatchResult objects.

Contains no football intelligence.

Depends only on:

```python
football_model.simulate_match(...)
```

This dependency inversion is intentional.

---

## CompetitionEngine

Classification

```
ACTIVE
```

Purpose

Resolve competitions using completed match results.

No prediction logic.

---

# Goal Sampling

## Dixon–Coles Hierarchical Sampler

Classification

```
ACTIVE
```

Current production scoreline sampler.

Receives:

```
lambda_home

lambda_away
```

Produces:

```
home goals

away goals
```

---

## Alternative Goal Samplers

Examples include:

- mixture samplers;
- volatility samplers;
- stochastic lambda samplers;
- bivariate samplers.

Classification

```
RESEARCH
```

Purpose

Benchmarking and experimental evaluation.

---

# Club Runtime

Current runtime:

```
Production repository

↓

Observation builder

↓

Production goal model

↓

FootballModelAdapter

↓

LeagueMatchSimulator
```

Classification

```
AUTHORITATIVE CLUB RUNTIME
```

Validated by:

- Study 073
- Study 074

---

# National-Team Runtime

Current runtime:

```
National-team repository

↓

National-team lambda model

↓

Configured scoreline sampler

↓

World Cup simulator
```

Classification

```
ACTIVE DOMAIN-SPECIFIC RUNTIME
```

This is intentionally separate from the club runtime.

Both share:

- scoreline-first philosophy;
- competition framework;
- scoreline sampling concepts.

They differ in:

- feature generation;
- repositories;
- rating priors;
- expected-goal models.

---

# World Cup Framework

Classification

```
ACTIVE
```

Purpose

Competition orchestration.

Responsibilities

- group stage;
- knockout stage;
- advancement;
- bracket resolution.

Football modeling is delegated.

---

# Outcome-First Engine

Legacy path:

```
Outcome probabilities

↓

Sample outcome

↓

Construct compatible scoreline
```

Classification

```
LEGACY
```

Still available only through explicit configuration.

Not the default runtime.

---

# Inference Package

```
inference/
```

Classification

```
LEGACY / COMPATIBILITY
```

Contains earlier production interfaces and prediction
infrastructure.

Should remain until a future archival review.

---

# ML Outcome Models

Examples:

```
LightGBM

Logistic Regression

Random Forest

XGBoost
```

Classification

```
RESEARCH
```

Purpose

Historical classifier development.

These models are valuable scientific assets but are no longer the
architectural center of the platform.

---

# Study Modules

```
research/studies/
```

Classification

```
RESEARCH
```

Responsibilities

- experimentation;
- validation;
- benchmarking;
- methodology.

Studies should not become runtime dependencies.

Version 2A successfully promoted reusable functionality from
Studies 073–084 into shared production modules.

---

# Evaluation Layer

Replay

↓

Performance Analysis

Classification

```
AUTHORITATIVE
```

Prediction generation and evaluation are intentionally separated.

Evaluation consumes frozen prediction artifacts.

Evaluation does not generate predictions.

---

# Current Public Runtime Hierarchy

Recommended conceptual hierarchy:

```
ProductionPredictionPipeline
        ↓
LiveMatchObservationBuilder
        ↓
ProductionGoalModel
        ↓
Frozen Goal Model Artifact
```

Competition simulation should instead depend on:

```
FootballModelAdapter
        ↓
simulate_match(...)
```

rather than directly invoking prediction internals.

---

# Runtime Status Summary

| Component | Status |
|-----------|--------|
| ProductionPredictionPipeline | AUTHORITATIVE |
| FootballModelAdapter | AUTHORITATIVE |
| ProductionClubRepositoryBuilder | AUTHORITATIVE |
| ProductionRepositorySchema | AUTHORITATIVE |
| ProductionClubRepository | ACTIVE |
| LiveMatchObservationBuilder | ACTIVE |
| ProductionGoalModel | ACTIVE |
| ClubEloRepository | ACTIVE |
| LeagueMatchSimulator | ACTIVE |
| CompetitionEngine | ACTIVE |
| Dixon–Coles Goal Sampler | ACTIVE |
| World Cup Simulator | ACTIVE DOMAIN-SPECIFIC |
| IntegratedClubGoalPredictor | COMPATIBILITY |
| inference/ | LEGACY / COMPATIBILITY |
| Outcome-first ML engine | LEGACY |
| Alternative goal samplers | RESEARCH |
| ML classifiers | RESEARCH |
| research/studies | RESEARCH |

---

# Version 2A Runtime Freeze

The Version 2A runtime is considered frozen.

Future work should:

- improve football intelligence;
- improve expected-goal prediction;
- improve calibration;
- improve player modeling;

without changing the established architectural boundaries unless
future evidence demonstrates that those boundaries are
insufficient.

The runtime classifications in this document should be updated
only when a component changes lifecycle status.