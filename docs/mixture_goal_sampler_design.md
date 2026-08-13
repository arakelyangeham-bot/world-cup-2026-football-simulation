mixture_goal_sampler_design.md

# Mixture Goal Sampler Design

## Problem

The calibrated Poisson goal model matches historical mean goals very well, but underestimates total-goal variance.

Historical total-goal variance:

7.285

Calibrated Poisson total-goal variance:

3.224

## Failed Simple Fixes

### Constant Negative Binomial

Improved total-goal variance, but inflated clean sheets and draw rates.

### Shared Match Tempo

Improved total-goal variance, but inflated draw rates because both teams' lambdas moved together.

## Design Requirements

A future sampler should:

1. Preserve calibrated mean goals.
2. Increase total-goal variance.
3. Avoid inflating draw rate.
4. Avoid inflating clean-sheet rate.
5. Preserve realistic high-scoring match frequency.

## Proposed Direction

Use a finite mixture sampler with latent match states.

Example states:

### Normal

Standard calibrated Poisson.

### Defensive

Suppress both teams slightly.

### Open

Increase both teams slightly.

### Chaotic-Asymmetric

Increase one team more than the other.

The key idea is that variance should come from heterogeneous match states, not from uniformly overdispersing every team score.

## Candidate State Structure

| State | Probability | Home Multiplier | Away Multiplier | Purpose |
|---|---:|---:|---:|---|
| Normal | 0.70 | 1.00 | 1.00 | preserve baseline |
| Defensive | 0.10 | 0.70 | 0.70 | low-scoring cagey games |
| Open | 0.10 | 1.25 | 1.25 | high-tempo games |
| Chaotic Home | 0.05 | 1.80 | 0.90 | asymmetric high home score |
| Chaotic Away | 0.05 | 0.90 | 1.80 | asymmetric high away score |

## Mean Preservation

Because multipliers can change average goals, the sampler should eventually include a mean-preserving correction.

For example:

average_multiplier =
sum(state_probability * state_multiplier)

corrected_lambda =
lambda / average_multiplier

This keeps expected goals approximately aligned with the calibrated model.

## Evaluation Metrics

The sampler should be evaluated against:

- average total goals
- total-goal variance
- draw rate
- clean-sheet rate
- five-plus-goal rate
- six-plus-goal rate
- score model fitness

## Next Implementation Step

Prototype a mean-preserving finite mixture sampler and benchmark it against:

- historical scores
- calibrated Poisson
- Negative Binomial
- shared-tempo sampler