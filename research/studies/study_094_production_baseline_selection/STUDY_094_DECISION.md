STUDY_094_DECISION

# Study 094 — Production Baseline Selection

## Status

**DECISION APPROVED**

## Decision

Promote `robust_zscore` to the default production
player-feature representation.

Retain `global_zscore` as the reproducible legacy baseline.

Do not remove or overwrite the Global artifacts.

## Decision scope

This decision applies to the current football-intelligence
architecture and its production artifact pipeline:

- player attribute scores;
- player role ratings;
- club representations;
- production-format club repositories;
- integrated Poisson club goal models;
- production prediction-pipeline construction.

The decision does not claim that Robust Z-score is universally
optimal across every league, season, competition, or future
feature family.

## Evidence reviewed

### Representation calibration

Studies 089–092 established that feature transformation is a
first-class configurable component of the football-intelligence
pipeline.

The three evaluated transformations were:

- `global_zscore`;
- `percentile_normal`;
- `robust_zscore`.

### Primary matched benchmark

Study 092C2B compared all three representations using:

- the same 306 Bundesliga fixtures;
- the same targets;
- the same fixture-date ClubElo priors;
- the same feature specification;
- the same chronological split;
- the same Poisson regularization;
- the same benchmark engine.

Result:

- preferred representation: `robust_zscore`;
- primary predictive metrics won by Robust;
- eight of ten comparison metrics won by Robust.

### Robustness benchmark

Study 092C2C evaluated:

- four chronological train fractions;
- four alpha values;
- 16 matched configurations;
- 48 total model fits.

Result:

- evidence classification: `STRONG`;
- configuration win rate: 87.5%;
- primary metric win rate: 92.0%;
- 103 primary-metric wins from 112 comparisons.

### Production artifact validation

Study 093A generated paired Global and Robust production
goal-model artifacts from the same 229-match training
population.

Both artifacts passed:

- artifact validation;
- JSON serialization;
- artifact loading;
- prediction round-trip reproduction;
- training-provenance validation.

### Operational production replay

Study 093B executed both candidates through the unchanged
production prediction pipeline.

Both candidates produced:

- 306 successful predictions;
- zero runtime failures;
- complete event-population preservation;
- valid expected goals;
- valid outcome probabilities.

### Production-path holdout evaluation

Study 093C evaluated the candidates on the same frozen
77-match chronological holdout.

Robust won all seven primary predictive metrics:

- combined Poisson deviance;
- combined goal MAE;
- total-goal MAE;
- goal-difference MAE;
- outcome log loss;
- outcome Brier score;
- exact-score log loss.

Overall metric result:

- Robust wins: 8;
- Global wins: 2.

## Promotion rationale

The evidence is consistent across:

1. representation calibration;
2. matched predictive benchmarking;
3. split and regularization robustness;
4. production artifact construction;
5. operational replay;
6. production-path holdout evaluation.

The production replay reproduced the benchmark ordering using
the real runtime path rather than a separate research-only
prediction implementation.

The observed gains are modest in absolute size but consistent
across all primary predictive metrics. Because the experimental
difference was isolated to the player-derived representation,
the evidence supports promotion of Robust Z-score.

## Methodological limitations

The current evidence is limited to:

- one Bundesliga season;
- a 77-match chronological holdout;
- retrospective static season-level player representations;
- fixture-date-valid ClubElo priors;
- the current integrated club-goal-model feature family.

The study does not establish:

- cross-league generalization;
- multi-season generalization;
- World Cup national-team generalization;
- prediction-date-frozen player-representation validity;
- universal superiority over future transformations.

## Production policy

### New default

```text
PLAYER_FEATURE_TRANSFORMATION = robust_zscore

or the equivalent configuration setting used by the player
attribute and rating builders.

Legacy baseline

global_zscore must remain available as:

a reproducible benchmark branch;
a rollback option;
a regression baseline;
a comparison representation for future studies.
Artifact policy

Canonical Robust artifacts should be created by copying or
promoting validated outputs, not by manually editing values.

Legacy Global artifacts must not be overwritten.

Every promoted artifact should retain:

transformation identity;
generation timestamp;
source dataset provenance;
feature specification;
model alpha;
training population;
training cutoff;
artifact version.
Required promotion work
Update the default feature-transformation configuration to
robust_zscore.
Generate canonical Robust player-attribute scores.
Generate canonical Robust player ratings.
Generate the canonical Robust club repository.
Generate the canonical Robust integrated goal-model
artifact.
Update production configuration paths to point to the
Robust repository and Robust model artifact.
Preserve all Global artifacts under explicit legacy names.
Run the full regression suite.
Run the canonical production replay.
Run World Cup tournament-simulation sensitivity before
the next formal release.
Required regression gates

The promotion is incomplete until all of the following pass:

player-intelligence test suite;
research integration tests;
repository loading tests;
production goal-model artifact tests;
live observation-builder tests;
production prediction-pipeline tests;
Bundesliga production replay;
tournament simulation smoke tests.
Rollback rule

Immediately restore Global Z-score as the production default if
the Robust promotion causes:

schema incompatibility;
repository population loss;
non-finite representation values;
model artifact loading failure;
replay runtime failures;
material regression in scoreline-distribution behavior;
tournament instability not explained by intended model
differences.
Final conclusion

Within the current evidence boundary, robust_zscore has
earned promotion from research candidate to production
baseline.

global_zscore remains the official legacy and rollback
baseline.