research_project_5_1_primary_match_generator_probability_engine.md

# Research Project 5.1 — Primary Match Generator Probability Engine

## Core Question

Can the existing production match generator become the primary prediction engine?

More specifically:

```text
Can we derive well-calibrated home/draw/away probabilities directly from the scoreline generator?
```

## Current Production Baseline

```text
Team Repository
  ↓
Feature Builder
  ↓
Calibrated LightGBM outcome model
  ↓
P(home win), P(draw), P(away win)
  ↓
Sample outcome
  ↓
Generate compatible scoreline
```

## Candidate Architecture

```text
Team Repository
  ↓
Calibrated expected-goals model
  ↓
Dixon-Coles hierarchical scoreline generator
  ↓
Scoreline distribution
  ↓
P(home win), P(draw), P(away win)
  ↓
Sample scoreline
  ↓
Derive outcome
```

## Research-Only Files

Create:

```text
research/match_generation/
  match_generator_probability_engine.py
  benchmark_match_generator_probability_engine.py
```

No production code changes yet.

## Candidate v1

Use the existing production components:

```text
LAMBDA_MODEL = calibrated
GOAL_SAMPLER = dixon_coles_hierarchical
tempo_cv = 0.60
team_cv = 0.10
rho = 0.30
```

## Probability Derivation

For each matchup:

```text
simulate or enumerate many scorelines
count:
  home_goals > away_goals → home_win
  home_goals = away_goals → draw
  home_goals < away_goals → away_win
normalize counts into probabilities
```

## Benchmark Against

The current production ML classifier:

```text
inference.MatchPredictor
```

## Required Metrics

```text
multiclass Brier score
multiclass log loss
accuracy
actual vs predicted distribution
reliability tables
expected calibration error
scoreline TVD
draw rate
mean total goals
tournament stability
```

## Promotion Rule

This candidate should not be promoted unless it:

```text
1. Matches or improves scoreline realism.
2. Matches or improves Brier score.
3. Matches or improves log loss.
4. Does not degrade draw calibration.
5. Preserves tournament stability.
6. Keeps the tournament simulator scoreline-driven.
```

## Expected Outcome

This experiment will tell us whether the production scoreline generator can move from supporting role to primary predictive engine.

If successful, the project architecture becomes:

```text
Football Intelligence
  ↓
Match Generator
  ↓
Scoreline
  ↓
Outcome
  ↓
Tournament
```
