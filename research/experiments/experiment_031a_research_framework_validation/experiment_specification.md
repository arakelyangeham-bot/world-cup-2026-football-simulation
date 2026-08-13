experiment_specification.md

# Experiment 031A — Research Framework Validation

## Research Question

Can the Version 3 research framework produce a valid experiment report from a completed synthetic competition?

## Motivation

Before comparing league and knockout formats in Experiment 031B, the project needs to validate the research pipeline itself.

Experiment 031A confirms that the following components work together:

- `Experiment`
- synthetic team strengths
- competition framework
- `ExperimentResult`
- `ExperimentRunner`
- metric library
- `ExperimentReport`

## Fixed Variables

- Synthetic 8-team strength ladder
- Single competition format
- Deterministic placeholder match results
- Metric set
- One experimental condition

## Independent Variable

None.

This is a framework validation experiment, not a comparative football experiment.

## Dependent Variables

- average champion strength
- strongest-team championship rate
- champion variance
- upset rate

## Hypothesis

The Version 3 research framework can convert completed synthetic competition results into a reusable metric report.

## Experimental Design

Use one synthetic 8-team league competition.

The league uses:

- 8 teams
- single round-robin
- deterministic placeholder results
- champion determined by standings

The output is converted into an `ExperimentResult`, evaluated by `ExperimentRunner`, and summarized through the metric library.

## Success Criteria

Experiment 031A succeeds if:

1. A synthetic competition run can be converted into `ExperimentRunResult`.
2. The run can be stored in `ExperimentResult`.
3. `ExperimentRunner` can evaluate all selected metrics.
4. An `ExperimentReport` is produced.
5. The report prints interpretable metric values.