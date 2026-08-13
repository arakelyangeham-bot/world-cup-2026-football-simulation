experiment_specification.md

# Experiment 031C — Production League vs Knockout

## Research Question

Does a league identify the strongest team more reliably than a knockout competition when using Production Football Model v1?

## Hypothesis

League competitions will still identify the strongest team more reliably than knockout competitions under the production scoreline-first match model.

## Football Model

Production Football Model v1:

- repository source: `dimension_specific`
- match engine: `production_scoreline_first`
- scoreline model: scoreline-first expected goals
- sampler: Dixon-Coles hierarchical sampler

## Experimental Conditions

### Condition A

- competition format: single round-robin league
- football model: Production Football Model v1

### Condition B

- competition format: seeded knockout
- football model: Production Football Model v1

## Metrics

- average champion strength
- strongest-team championship rate
- champion variance
- upset rate

## Purpose

This is a replication of Experiment 031B using the production football model instead of the synthetic match model.