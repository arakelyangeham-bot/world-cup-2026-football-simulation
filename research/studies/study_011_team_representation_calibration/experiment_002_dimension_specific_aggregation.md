experiment_002_dimension_specific_aggregation.md

# Experiment 002 — Dimension-Specific Aggregation

## Objective

Test whether different football dimensions should use different aggregation strategies.

## Research Question

Should attack, midfield, defense, and goalkeeper be aggregated differently?

## Hypothesis

Different football dimensions behave differently:

- attack may be star-driven,
- midfield may reward depth and balance,
- defense may require collective structure,
- goalkeeper may depend mostly on the best available player.

## Candidate Design

```text
attack      → star_weighted
midfield    → starter_plus_depth
defense     → top_11_mean
goalkeeper  → best_player

Evaluation Metrics
dynamic range by dimension
repository comparison against legacy
scoreline realism
tournament realism
Non-Goals

No simulator changes.

No match-engine changes.