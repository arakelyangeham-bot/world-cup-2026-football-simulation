scoreline_first_research_plan.md

# Scoreline-First Research Plan v1

## Purpose

This research track evaluates whether the simulator should become primarily scoreline-driven.

The current production system uses an outcome-first hybrid:

```text
ML outcome probabilities
  ↓
sample W/D/L outcome
  ↓
generate compatible scoreline
```

This research track tests a scoreline-first alternative:

```text
team features
  ↓
scoreline distribution
  ↓
sample scoreline
  ↓
derive W/D/L outcome
```

## Core Research Question

Can a scoreline-first model match or beat the current production outcome-first hybrid on both:

```text
scoreline realism
outcome calibration
```

## Motivation

Football results are naturally scoreline events.

A match does not first become a home win, draw, or away win. It becomes a scoreline:

```text
2-1
0-0
1-1
0-2
```

The outcome is derived afterward.

Therefore, the cleaner long-term architecture is:

```text
scoreline distribution
  ↓
outcome probabilities
  ↓
competition simulation
```

## Production Baseline

The baseline remains the current production hybrid:

```text
Calibrated ML outcome model
  ↓
sample desired outcome
  ↓
Dixon-Coles / configured scoreline generator
  ↓
compatible scoreline
```

This baseline should not be replaced unless a candidate performs better or equivalently across all required benchmarks.

## Candidate v1

The first scoreline-first candidate should be intentionally simple:

```text
calibrated expected-goals model
  +
Dixon-Coles-style scoreline probability adjustment
  +
scoreline distribution enumeration
```

The candidate should produce a full probability table:

```text
home_goals
away_goals
scoreline_probability
```

Then derive:

```text
home_win_probability = sum(P(h, a) where h > a)
draw_probability     = sum(P(h, a) where h = a)
away_win_probability = sum(P(h, a) where h < a)
```

## Research-Only Implementation

No production code should change initially.

Suggested location:

```text
research/scoreline_first/
  scoreline_distribution_model.py
  benchmark_scoreline_first.py
```

The production simulator should remain untouched during this research phase.

## Required Metrics

The candidate must be evaluated against the current production baseline using:

```text
scoreline TVD
multiclass Brier score
multiclass log loss
expected calibration error
home/draw/away reliability tables
draw rate
mean total goals
0-0 frequency
1-1 frequency
1-0 frequency
2-1 frequency
tournament probability stability
```

## Promotion Criteria

A scoreline-first candidate may be considered for production only if it:

```text
1. Improves or preserves scoreline realism.
2. Matches or improves multiclass Brier score.
3. Matches or improves multiclass log loss.
4. Does not degrade draw calibration.
5. Does not destabilize tournament probabilities without explanation.
6. Preserves clean separation from tournament simulation code.
```

## Non-Goals for v1

Do not begin with:

```text
neural scoreline models
player-level scoreline modeling
tactical interaction models
manual scoreline overrides
production simulator rewrites
```

The first goal is to test the architecture, not maximize complexity.

## Expected Outcome

This research track should answer whether the project should eventually move from:

```text
outcome-first simulation
```

to:

```text
scoreline-first simulation
```

If the scoreline-first candidate succeeds, the simulator becomes more aligned with the project’s core purpose: predicting scores and letting those scores determine tournament results.
