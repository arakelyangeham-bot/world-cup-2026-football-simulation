simulation_comparison_plan.md

# Simulation Comparison Plan

## Purpose

Evaluate candidate team-aggregation strategies through the full simulation pipeline.

## Candidate Strategies

- starter_plus_depth
- top_11_mean
- top_5_mean
- star_weighted

Uniform mean is excluded because it produced excessive compression.

## Evaluation Pipeline

```text
PlayerRepresentations
        ↓
Aggregation Strategy
        ↓
Player Intelligence Team Repository
        ↓
Scoreline-First Match Engine
        ↓
Monte Carlo Tournament Simulation
        ↓
Simulation Statistics

Metrics

Compare each strategy by:

average goals per match
draw rate
extra-time frequency
penalty frequency
champion distribution
semifinal distribution
scoreline realism
dynamic range of team representation
Guardrail

Do not replace the production repository yet.

Each aggregation strategy should write to a separate sandbox output path.

Success Criteria

A strategy is promising if it:

improves team separation,
preserves football realism,
avoids implausible tournament dominance,
remains interpretable,
improves or matches scoreline calibration.
Next Engineering Task

Build a sandbox repository generator:

scripts/build_player_intelligence_repository_by_strategy.py

This script should output one repository per aggregation strategy.