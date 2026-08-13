experiment_003_tournament_behavior_comparison.md

# Experiment 003 — Tournament Behavior Comparison

## Objective

Evaluate whether Player Intelligence aggregation strategies improve full tournament behavior, not only scoreline realism.

## Motivation

The scoreline benchmark showed that several Player Intelligence repositories matched or outperformed the legacy repository on scoreline TVD.

The next question is whether those improvements persist when the repositories are used inside the full World Cup tournament simulator.

## Candidate Repositories

Evaluate:

- legacy
- dimension_specific
- star_weighted
- top_11_mean

Exclude for now:

- starter_plus_depth

Reason:

It produced the weakest scoreline behavior among Player Intelligence strategies.

## Research Question

Do Player Intelligence repositories produce more realistic tournament behavior than the legacy repository?

## Metrics

Compare:

- champion probabilities
- runner-up probabilities
- semifinal probabilities
- average goals
- draw rate
- scoreline realism
- upset behavior
- concentration of winners
- plausibility of top contenders

## Non-Goals

This experiment will not:

- retrain the Poisson model
- modify the match engine
- modify the goal sampler
- alter tournament structure

## Success Criteria

A candidate strategy is promising if it:

- preserves or improves scoreline realism,
- produces plausible tournament distributions,
- avoids excessive dominance by a small number of teams,
- avoids excessive randomness,
- remains interpretable.

## Engineering Task

Create:

```text
scripts/run_strategy_tournament_benchmark.py

his script should run the tournament simulator once per selected repository and write outputs under:

outputs/study_011_team_representation_calibration/tournament_benchmarks/
Guiding Principle

A team representation should be judged not only by match-level realism, but also by the tournament worlds it creates.