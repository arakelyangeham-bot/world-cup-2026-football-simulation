hierarchical_stochastic_lambda_design.md

# Hierarchical Stochastic Lambda Sampler Design

## Motivation

The calibrated Poisson model preserves historical mean goals extremely well, but underestimates total-goal variance.

Exploratory sampler results:

- Negative Binomial improved variance but inflated clean sheets.
- Shared tempo improved variance but inflated draws.
- Fixed mixture samplers produced only small improvements.
- Stochastic lambda produced the best variance improvement while preserving mean goals.

## Core Idea

Instead of treating lambda as fixed, model each match as having stochastic expected goals.

A hierarchical version separates uncertainty into:

1. Shared match tempo
2. Home-team finishing variation
3. Away-team finishing variation

## Model

Base calibrated lambdas:

```text
lambda_home
lambda_away