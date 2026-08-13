experiment_002_competition_aware_representation.md

# Experiment 002 — Competition-Aware Player Representation

## Objective

Investigate whether player evidence should be weighted differently depending on competition source.

## Research Question

Should 1,000 minutes in one competition count the same as 1,000 minutes in another?

## Motivation

Experiment 001 showed that evidence confidence and minutes materially affect player representation.

However, those measures mostly answer:

```text
How much evidence do we have?

Experiment 002 asks:

Where did the evidence come from?
Hypothesis

Player evidence should be competition-aware.

Minutes from stronger or more relevant competitions should contribute more to player representation than minutes from weaker or less relevant competitions.

Candidate Strategies
Strategy A — Current

Use existing role ratings as-is.

Strategy B — Competition Count

Reward players observed across more competitions.

Strategy C — Competition Importance

Weight evidence by competition importance.

Strategy D — Source Diversity

Reward players with evidence from multiple competition types.

Strategy E — Combined Competition Evidence

Combine competition count, importance, and weighted evidence.

Inputs
player_ratings.csv
source_competitions
competition_count
season_count
total_weighted_evidence
evidence_confidence
Metrics

For each strategy measure:

player rating spread
team representation spread
evidence weight distribution
top-team sensitivity
downstream scoreline realism
Non-Goals

No changes to:

match engine
goal sampler
tournament simulator
Poisson coefficients
Success Criteria

A competition-aware representation is promising if it:

remains interpretable,
preserves meaningful variation,
avoids over-compressing ratings,
improves team representation quality.
Guiding Principle

Player evidence should reflect both quantity and source quality.