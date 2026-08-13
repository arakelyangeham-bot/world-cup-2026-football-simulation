Club_Goal_Model_v1

# Integrated Club Goal Model

## Version

**1.0**

## Status

**Recommended research baseline**

## Feature specification

`attack_defense_attack_depth_rating_prior`

## Purpose

The Integrated Club Goal Model v1 is the recommended
reference configuration for future club-football goal-model
experiments.

It combines two distinct sources of football information:

1. Player-derived team intelligence
2. Historical performance-derived club strength

## Model inputs

### Home-goal model

- `home_attack`
- `away_defense`
- `attack_depth_diff`
- `rating_prior_diff`

### Away-goal model

- `away_attack`
- `home_defense`
- `attack_depth_diff`
- `rating_prior_diff`

## Information sources

### Player-derived representation

The player-intelligence pipeline provides:

- attacking strength;
- defensive strength;
- attacking depth.

These features are derived from historically valid team
representations rather than future or same-season information.

### Historical rating prior

ClubElo provides a temporally valid historical club-strength
rating for the match prediction date.

The match-level prior is represented as:

`rating_prior_diff = home_rating_prior - away_rating_prior`

## Evidence chain

### Study 052 — Feature Ablation

Attack depth was identified as the strongest useful extension
to the attack-and-defense baseline.

The study also found that larger feature combinations could
overfit and that several intuitively plausible features did
not consistently improve prediction.

### Study 054 — Attack-Depth Stability

Attack depth was evaluated across multiple chronological
training fractions and regularization values.

It improved combined Poisson deviance in most configurations,
particularly once sufficient historical training data were
available.

### Study 060 — ClubElo Observation Enrichment

Historical ClubElo ratings were added to the validated
full-squad observation dataset.

The enrichment preserved the full match population and passed:

- event-population validation;
- team-name agreement;
- match-date agreement;
- rating-difference arithmetic;
- ClubElo provenance;
- temporal-validity checks.

### Study 061 — Incremental Information Benchmark

ClubElo was tested through controlled paired comparisons.

Adding `rating_prior_diff` improved the attack-and-defense
baseline across every tested chronological split and
regularization value for the principal goal and scoreline
metrics.

ClubElo also remained useful after attack depth was included.

The information-overlap analysis found that player-derived
attack, defense, and attack-depth differences explained only
part of the variance in ClubElo. Therefore, ClubElo contained
substantial information that was not recoverable from the
player-derived representation alone.

## Excluded features

The Version 1 baseline does not include:

- midfield difference;
- goalkeeper difference;
- midfield depth;
- defensive depth;
- aggregate squad quality.

These features remain available for research, but they did
not earn inclusion in the recommended baseline through the
completed controlled studies.

## Research policy

Future club goal-model improvement studies should normally:

1. Use this specification as the baseline.
2. Add one clearly defined football concept at a time.
3. Preserve matched observation populations.
4. Use chronological train/test splits.
5. Compare candidates across multiple regularization values.
6. Report both predictive performance and coefficient
   stability.
7. Promote a new baseline only after consistent evidence
   across controlled experiments.

## Versioning policy

Minor version changes may be used when an additional validated
feature extends the same general modeling framework.

Major version changes should be reserved for substantial
changes such as:

- a new goal-distribution family;
- dynamic lineup integration;
- a fundamentally different rating-prior system;
- hierarchical or league-specific modeling;
- major changes to the training-data protocol.

## Current recommendation

The project's current club goal-model reference baseline is:

`attack_defense_attack_depth_rating_prior`

This recommendation records the current evidence. It does not
prevent exploratory research or imply that the model is final.