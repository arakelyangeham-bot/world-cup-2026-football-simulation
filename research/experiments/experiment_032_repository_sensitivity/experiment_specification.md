experiment_specification.md

# Experiment 032 — Repository Sensitivity Analysis

## Research Question

How sensitive are competition outcomes to the choice of team representation repository?

## Motivation

Version 2 produced multiple team representation repositories:

- `legacy`
- `dimension_specific`
- `top_11_mean`
- `top_5_mean`
- `star_weighted`
- `starter_plus_depth`

These repositories were originally engineering alternatives for improving the simulator.

Version 3 treats them as scientific variables.

Experiment 032 investigates whether changing the team representation repository materially changes competition outcomes.

## Fixed Variables

- team set
- competition formats
- production scoreline-first match engine
- goal sampler
- simulation count
- random seed policy
- metrics
- experiment runner

## Independent Variable

Team repository source.

## Conditions

Each repository source is tested under the same competition setup.

Repositories:

1. `legacy`
2. `dimension_specific`
3. `top_11_mean`
4. `top_5_mean`
5. `star_weighted`
6. `starter_plus_depth`

## Dependent Variables

- average champion strength
- strongest-team championship rate
- champion variance
- upset rate

## Experimental Design

Repeat the League vs Knockout comparison from Experiment 031C across all repository sources.

For each repository:

1. Run league condition.
2. Run knockout condition.
3. Compute the same metric set.
4. Compare metric changes across repositories.

## Hypothesis

Repository choice will change the magnitude of competition outcomes, but the qualitative conclusion from Experiment 031 should remain stable: league competitions will identify the strongest team more reliably than knockout competitions.

## Success Criteria

Experiment 032 succeeds if it produces a comparison table showing how league and knockout outcomes vary across repository sources.