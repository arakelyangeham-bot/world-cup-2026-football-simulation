production_football_model_v1.md

# Production Football Model v1

## Purpose

Production Football Model v1 defines the canonical football model used for early Version 3 computational football experiments.

Its purpose is to provide a stable baseline so that experiments can change one research variable at a time without repeatedly redefining the underlying football model.

## Motivation

Experiment 031B used a synthetic match model to validate the Version 3 research framework.

Experiment 031C will repeat the League vs Knockout question using the production football model.

Before doing that, the project needs a clear definition of what “production football model” means.

## Model Definition

Production Football Model v1 consists of:

- repository source: `dimension_specific`
- match engine: `production_scoreline_first`
- scoreline generation: scoreline-first match simulation
- goal model: production expected-goals / Poisson-based model
- team strength source: Player Intelligence-derived team repository
- competition layer: Competition Framework v1
- research layer: Version 3 Research Framework

## Repository Source

The default repository source is:

```text
dimension_specific

This is selected because it was the strongest production candidate from the Player Intelligence v1 evaluation.

It balances:

interpretability,
scoreline realism,
plausible tournament behavior,
and modular compatibility with the simulator.
Match Engine

The default match engine is:

production_scoreline_first

This refers to the scoreline-first match simulation architecture developed during the production simulator phase.

The model should use the same match simulation logic used by the current production World Cup simulator unless a later experiment explicitly changes the match engine.

Fixed Baseline Assumption

In Version 3 experiments using Production Football Model v1, the following should remain fixed unless explicitly stated:

repository source
match engine
goal model
scoreline generation method
team strength interpretation
random seed policy
metric definitions
Experimental Use

A Version 3 experiment may define conditions such as:

Condition A:
competition_format = league
football_model = Production Football Model v1

Condition B:
competition_format = knockout
football_model = Production Football Model v1

In this setup, the independent variable is competition format.

The football model remains fixed.

Relationship to Experiment 031C

Experiment 031C will use Production Football Model v1 to repeat the League vs Knockout comparison from Experiment 031B under a more realistic football model.

The goal is not to ask a new research question.

The goal is to test whether the Experiment 031B conclusion is robust when the synthetic match model is replaced by the production football model.

Future Versions

Possible future football model versions include:

Production Football Model v2

May include improved lineup uncertainty, injuries, or updated Player Intelligence.

Production Football Model v3

May include chemistry, tactical fit, or dynamic form.

Repository Comparison Models

Future experiments may intentionally vary repository source:

legacy
dimension_specific
top_11_mean
top_5_mean
star_weighted
starter_plus_depth

Those should be treated as separate experimental conditions, not silent changes to Production Football Model v1.

Status

Production Football Model v1 is the default football baseline for early Version 3 research experiments.