team_strength_representation_study.md

# Team Strength Representation Study

## Purpose

Investigate whether the current player-derived team-strength representation contains enough variation to support realistic scoreline generation.

## Motivation

Recent calibration audits showed that the scoreline-first engine and Poisson goal model are functioning correctly, but the runtime team repository produces very compressed Poisson features.

The likely bottleneck is now the team-strength representation, not the match engine.

## Key Finding

Current repository features:

```text
poisson_attack  ≈ 1.03 with very low spread
poisson_defense ≈ 0.97 with very low spread

This gives the goal model little information beyond FIFA points.

Research Question

What should a team-strength representation preserve?

Candidate Requirements

A useful team-strength representation should preserve:

attacking quality
defensive solidity
goalkeeper quality
squad depth
starting XI quality
player availability
role balance
variation between elite and weaker teams
compatibility with the expected-goals model
Current Limitation

The current aggregation method produces Poisson features with low dynamic range.

This may cause the goal model to rely too heavily on FIFA points.

Investigation Plan
Audit current feature spread.
Compare composite ratings vs Poisson-specific ratings.
Compare full-squad aggregation vs expected-lineup aggregation.
Identify which features preserve meaningful team separation.
Propose a revised team-strength representation.
Non-Goals

This study will not modify the match engine.

This study will not tune goal-sampler parameters.

This study will not change tournament logic.

Success Criteria

The study succeeds if it identifies a better representation candidate that:

has meaningful variation,
remains interpretable,
can be generated from player data,
maps cleanly into the Team Repository,
can be benchmarked against the current representation.
Guiding Principle

Do not tune the simulator to compensate for weak representations.

Improve the representation first.