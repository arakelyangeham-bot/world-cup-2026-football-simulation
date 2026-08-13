synthetic_benchmark_specification

# Study 089A — Synthetic Aggregation Benchmark Specification

## 1. Purpose

This document defines the synthetic benchmark used to evaluate candidate
team-representation aggregation functions.

The benchmark converts the mathematical axioms and football behavioral
properties established in `aggregation_axioms.md` into explicit,
repeatable experiments.

The benchmark must be completed before any candidate aggregator is
evaluated on Bundesliga clubs or entered into a goal-model benchmark.

Its purpose is to answer:

> Does the aggregation behave mathematically and footballistically as
> intended under controlled player populations?

Synthetic scenarios isolate aggregation behavior from:

- player-data quality;
- identity resolution;
- club strength;
- league effects;
- goal-model specification;
- match-result noise.

---

# 2. Benchmark Scope

The synthetic benchmark operates on one-dimensional player projections.

For each scenario, a population is defined as:

```text
P = {p₁, p₂, ..., pₙ}
```

where every `pᵢ` is a normalized player projection.

Initial benchmark values should lie in:

```text
[0, 1]
```

Candidate aggregation methods map the player population to:

```text
A(P)
```

The benchmark initially evaluates primary strength and depth functions
separately.

Role-aware and structural aggregators may later consume richer synthetic
player records.

---

# 3. Candidate Aggregation Families

The first synthetic benchmark should include the following methods.

## Control 1 — Arithmetic top-five mean

```text
mean(top 5)
```

This is the current primary-strength baseline.

---

## Control 2 — Whole-population mean

```text
mean(all players)
```

This represents the current depth-style baseline.

---

## Candidate 1 — Mild rank-weighted top five

Weights:

```text
0.24
0.22
0.20
0.18
0.16
```

---

## Candidate 2 — Moderate rank-weighted top five

Weights:

```text
0.30
0.25
0.20
0.15
0.10
```

---

## Candidate 3 — Strong rank-weighted top five

Weights:

```text
0.40
0.25
0.15
0.12
0.08
```

---

## Candidate 4 — Star-influence aggregation

For:

```text
α ∈ {0.10, 0.20, 0.30}
```

compute:

```text
(1 - α) × mean(top 5)
+
α × max(top 5)
```

---

## Candidate 5 — Power mean

For:

```text
p ∈ {1.25, 1.50, 2.00}
```

compute the power mean of the top five projections.

---

## Candidate 6 — Softmax-weighted top five

For a small predefined set of concentration parameters:

```text
β ∈ {1, 3, 5}
```

The implementation must document the input scale because softmax
sensitivity depends on that scale.

---

## Candidate 7 — Replacement-drop-off depth

Compute:

```text
primary mean
=
mean(ranks 1–5)
```

```text
replacement mean
=
mean(ranks 6–10)
```

```text
replacement drop-off
=
primary mean - replacement mean
```

---

## Candidate 8 — Distribution-shape augmentation

Retain the top-five mean and report:

- maximum;
- minimum of the top five;
- range;
- standard deviation;
- star gap;
- concentration.

These are feature augmentations rather than scalar replacements.

---

# 4. Common Input Contract

Every scalar aggregation function should accept:

```python
Sequence[float]
```

and return:

```python
float
```

Every function must:

- reject an empty population;
- reject non-finite values;
- document minimum population size;
- sort internally when rank ordering is required;
- avoid mutating the input;
- produce deterministic output;
- validate required parameters.

Functions that require at least five or ten players must fail explicitly
when the population is too small.

Silent padding is not permitted.

---

# 5. Common Output Contract

Each benchmark row should record:

```text
scenario_id
scenario_family
scenario_description
aggregator_id
aggregator_family
parameterization
baseline_value
modified_value
absolute_delta
relative_delta
expected_direction
observed_direction
axiom_or_property
pass
notes
```

For scenarios that compare two independent populations rather than a
baseline and modification, use:

```text
population_a_value
population_b_value
difference_b_minus_a
```

as additional fields.

---

# 6. Numerical Tolerance

Deterministic equality tests should use a documented floating-point
tolerance.

Recommended default:

```text
absolute tolerance = 1e-12
relative tolerance = 1e-12
```

Behavioral scenarios should not use the same tolerance as mathematical
identity checks.

Their thresholds must be defined separately.

---

# 7. Mandatory Axiom Tests

## Test A1 — Determinism

### Procedure

Evaluate the same population repeatedly.

Example:

```text
0.91
0.87
0.84
0.79
0.75
0.70
0.66
```

Run each aggregation at least ten times.

### Expected result

All outputs must be identical within numerical tolerance.

### Pass rule

```text
maximum output - minimum output ≤ tolerance
```

---

## Test A2 — Permutation invariance

### Procedure

Evaluate a population in:

- descending order;
- ascending order;
- a fixed shuffled order;
- several seeded random permutations.

### Expected result

Every permutation must produce the same result.

### Pass rule

All outputs equal the canonical result within numerical tolerance.

---

## Test A3 — Monotonicity

### Procedure

Begin with:

```text
0.90
0.85
0.80
0.75
0.70
```

Increase one player at a time by:

```text
0.01
```

Repeat at each rank.

### Expected result

The aggregation must never decrease.

### Pass rule

```text
modified_value ≥ baseline_value - tolerance
```

For top-k functions, increasing a player outside the top-k may produce
zero change. That is still monotonic.

---

## Test A4 — Continuity

### Procedure

Perturb one player by:

```text
±0.000001
±0.0001
±0.01
```

Test:

- highest-ranked player;
- fifth-ranked player;
- threshold player near rank five;
- player outside the primary top-k.

### Expected result

Small perturbations produce correspondingly small output changes.

### Report

Record local sensitivity:

```text
output_delta / input_delta
```

### Pass rule

No discontinuous jump beyond the amount implied by a documented
rank-boundary change.

Rank-based methods should receive special threshold diagnostics rather
than automatic failure.

---

## Test A5 — Boundedness

### Procedure

Use several populations in:

```text
[0, 1]
```

including:

```text
all zeros
all ones
mixed extremes
random values
```

### Expected result

For mean-like scalar strength aggregators:

```text
min(P) ≤ A(P) ≤ max(P)
```

### Exception

Derived features such as:

- range;
- standard deviation;
- replacement drop-off;

have their own valid ranges and should be checked separately.

---

## Test A6 — Identity

### Procedure

For:

```text
x ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

construct populations where every player equals `x`.

### Expected result

Every mean-like scalar aggregator returns exactly `x`.

Distribution-shape features should return:

```text
range = 0
standard deviation = 0
star gap = 0
```

---

## Test A7 — Symmetry

### Procedure

Construct populations containing repeated equal values and exchange
their positions and identifiers.

### Expected result

Equal-valued players must be treated identically.

No output may depend on arbitrary identity fields.

---

# 8. Stability Scenarios

## Scenario S1 — Nearly identical starter replacement

Baseline:

```text
0.92
0.88
0.84
0.80
0.76
```

Modified:

```text
0.92
0.88
0.84
0.80
0.75
```

### Football interpretation

A marginal downgrade in the weakest primary contributor.

### Expected behavior

Small negative delta.

---

## Scenario S2 — Nearly identical elite replacement

Baseline:

```text
0.92
0.88
0.84
0.80
0.76
```

Modified:

```text
0.91
0.88
0.84
0.80
0.76
```

### Football interpretation

A small downgrade in the strongest player.

### Expected behavior

Small negative delta.

Rank-weighted, star-influence, power-mean, and softmax methods may react
more strongly than the arithmetic mean.

---

## Scenario S3 — Threshold swap

Baseline:

```text
0.90
0.85
0.80
0.75
0.7001
0.7000
```

Modified:

```text
0.90
0.85
0.80
0.75
0.7000
0.7001
```

### Football interpretation

Two nearly identical players exchange fifth and sixth rank.

### Expected behavior

Negligible or zero change.

This scenario is especially important for rank-based methods.

---

# 9. Responsiveness Scenarios

## Scenario R1 — Elite player removal

Baseline:

```text
0.98
0.86
0.84
0.82
0.80
0.76
```

Modified:

```text
0.86
0.84
0.82
0.80
0.76
```

### Football interpretation

The elite player becomes unavailable.

### Expected behavior

Meaningful negative delta.

---

## Scenario R2 — Average player removal

Baseline:

```text
0.98
0.86
0.84
0.82
0.80
0.76
```

Modified:

```text
0.98
0.86
0.84
0.82
0.76
```

### Football interpretation

A non-elite primary contributor becomes unavailable.

### Expected behavior

Negative delta smaller than Scenario R1 for methods claiming elite
responsiveness.

---

## Scenario R3 — Superstar addition

Baseline:

```text
0.84
0.83
0.82
0.81
0.80
```

Modified:

```text
0.99
0.84
0.83
0.82
0.81
0.80
```

### Expected behavior

Positive delta.

Star-sensitive methods should exceed the baseline arithmetic response.

---

## Scenario R4 — Weak fringe addition

Baseline:

```text
0.90
0.86
0.82
0.78
0.74
0.70
0.66
0.62
0.58
0.54
```

Modified:

```text
0.90
0.86
0.82
0.78
0.74
0.70
0.66
0.62
0.58
0.54
0.20
```

### Expected behavior

Primary top-five strength should not change.

Whole-population depth may decline.

Replacement-drop-off depth should remain unchanged if the added player
falls outside the replacement group.

---

# 10. Distribution-Awareness Scenarios

## Scenario D1 — Balanced versus top-heavy

Population A:

```text
0.95
0.90
0.85
0.80
0.75
```

Population B:

```text
0.85
0.85
0.85
0.85
0.85
```

Both arithmetic means equal:

```text
0.85
```

### Expected behavior

Arithmetic mean:

```text
equal
```

Star-sensitive and distribution-aware methods:

```text
different
```

The benchmark does not presume which population is better.

It records the encoded hypothesis.

---

## Scenario D2 — Extreme superstar versus balanced quality

Population A:

```text
0.99
0.70
0.70
0.70
0.70
```

Population B:

```text
0.758
0.758
0.758
0.758
0.758
```

Both arithmetic means equal:

```text
0.758
```

### Expected behavior

The arithmetic mean treats the populations equally.

Other methods reveal how strongly they reward concentrated elite
quality.

---

## Scenario D3 — Strong core with steep drop-off

Population A:

```text
0.92
0.90
0.88
0.86
0.84
0.60
0.58
0.56
0.54
0.52
```

Population B:

```text
0.84
0.83
0.82
0.81
0.80
0.79
0.78
0.77
0.76
0.75
```

### Football interpretation

Population A has a stronger first unit and weak replacements.

Population B is more balanced.

### Expected behavior

Primary strength favors Population A.

Depth and drop-off features should reveal Population B's superior
replacement structure.

---

# 11. Depth Scenarios

## Scenario P1 — Roster-size dilution

Baseline:

```text
0.90
0.88
0.86
0.84
0.82
0.78
0.76
0.74
0.72
0.70
```

Modified:

```text
0.90
0.88
0.86
0.84
0.82
0.78
0.76
0.74
0.72
0.70
0.30
0.25
0.20
0.15
0.10
```

### Expected behavior

Top-five strength:

```text
unchanged
```

Whole-population mean:

```text
decreases materially
```

Replacement ranks 6–10:

```text
unchanged
```

### Interpretation

This scenario tests whether a depth measure confuses fringe-roster size
with usable replacement quality.

---

## Scenario P2 — Replacement-unit improvement

Baseline:

```text
0.90
0.88
0.86
0.84
0.82
0.65
0.63
0.61
0.59
0.57
```

Modified:

```text
0.90
0.88
0.86
0.84
0.82
0.75
0.73
0.71
0.69
0.67
```

### Expected behavior

Primary strength:

```text
unchanged
```

Replacement mean:

```text
increases
```

Replacement drop-off:

```text
decreases
```

---

## Scenario P3 — Starter and replacement improvement

Improve every player by:

```text
0.02
```

subject to an upper bound of:

```text
1.00
```

### Expected behavior

Primary strength and replacement quality both increase smoothly.

Drop-off may remain unchanged.

---

# 12. Rank-Boundary Scenarios

## Scenario B1 — Fifth/sixth crossing

Baseline:

```text
0.90
0.85
0.80
0.75
0.7000
0.6999
```

Modified:

```text
0.90
0.85
0.80
0.75
0.7000
0.7001
```

### Expected behavior

Top-five methods change only by the tiny crossing difference.

No large jump is acceptable.

---

## Scenario B2 — Tied ranks

Population:

```text
0.90
0.85
0.80
0.75
0.70
0.70
0.70
```

### Expected behavior

Repeated runs and permutations produce identical output.

---

## Scenario B3 — Multiple simultaneous crossings

Construct several values near the fifth-rank threshold and perturb them
slightly.

### Expected behavior

Output remains stable relative to the perturbation size.

---

# 13. Scale Scenarios

## Scenario C1 — Uniform shift

Baseline:

```text
0.80
0.78
0.76
0.74
0.72
```

Modified:

```text
0.81
0.79
0.77
0.75
0.73
```

### Expected behavior

Translation-equivariant methods should increase by:

```text
0.01
```

Softmax and other scale-sensitive methods should be audited explicitly.

---

## Scenario C2 — Uniform multiplicative change

Multiply all values by:

```text
1.05
```

while preserving the valid scale.

### Expected behavior

Record whether relative player importance is preserved.

---

## Scenario C3 — Alternative numeric scale

Represent equivalent player quality on:

```text
[0, 1]
```

and:

```text
[0, 100]
```

### Expected behavior

After appropriate normalization, methods should encode the same ranking
and football conclusion.

Softmax methods must document their dependence on scale and parameter
rescaling.

---

# 14. Structural Scenarios

The first benchmark is primarily one-dimensional.

Structural scenarios require synthetic player records with role labels.

Recommended schema:

```text
player_id
primary_role
attack
midfield
defense
goalkeeper
```

## Scenario T1 — Balanced XI

Construct a plausible role-balanced eleven.

---

## Scenario T2 — No natural defenders

Replace natural defenders with high-rated attacking players.

### Expected behavior

Pure dimensional top-k methods may not detect the structural problem.

Role-balance features should.

---

## Scenario T3 — Missing striker

Construct a strong XI with no natural striker.

### Expected behavior

General attack strength may remain high.

Striker-coverage features should show the deficiency.

---

## Scenario T4 — Fallback-heavy lineup

Fill several slots using compatible but non-exact roles.

### Expected behavior

Fallback-role count and structural-balance features should deteriorate.

These scenarios are exploratory and should not block the first scalar
aggregation benchmark.

---

# 15. Comparative Behavioral Metrics

For each aggregator, compute:

## Stability score

Average absolute delta under small-perturbation scenarios.

Lower is more stable.

---

## Elite responsiveness score

Delta under elite removal divided by delta under comparable ordinary
replacement.

Example:

```text
elite_removal_delta
/
ordinary_replacement_delta
```

Higher values indicate greater elite sensitivity.

---

## Fringe sensitivity score

Absolute change in primary strength after weak fringe additions.

Lower is preferable.

---

## Distribution separation score

Difference between equal-mean balanced and top-heavy populations.

This is descriptive, not automatically good or bad.

---

## Rank-boundary sensitivity

Maximum output delta produced by a tiny threshold crossing.

Lower is preferable.

---

## Depth validity score

Change in depth under replacement improvement compared with change under
fringe-roster expansion.

A useful depth measure should react more strongly to the former.

---

# 16. Pass and Failure Philosophy

Mandatory axioms should receive binary pass/fail results.

Football behavioral properties should not usually receive simplistic
binary judgments.

Instead, they should report:

- direction;
- magnitude;
- comparison with control;
- interpretation.

For example, rewarding a superstar more strongly is not universally a
pass.

It is evidence about the hypothesis encoded by the aggregator.

---

# 17. Recommended Outputs

The implementation study should produce:

```text
synthetic_populations.csv
```

One row per player per scenario population.

---

```text
aggregation_scenario_results.csv
```

One row per aggregator-scenario comparison.

---

```text
aggregation_axiom_results.csv
```

One row per aggregator and mandatory axiom test.

---

```text
aggregation_behavior_metrics.csv
```

One summary row per aggregator.

---

```text
aggregation_parameter_registry.csv
```

Exact parameters for every tested specification.

---

```text
aggregation_rankings.csv
```

Comparative rankings by stability, responsiveness, fringe sensitivity,
and depth validity.

---

```text
study_089b_metadata.json
```

Study metadata and boundaries.

---

```text
study_089b_report.md
```

Human-readable interpretation.

---

# 18. Required Metadata

The metadata file should record:

```text
study_id
study_name
generated_at
status
aggregation_specification_count
scenario_count
axiom_test_count
random_seed
floating_point_tolerances
input_scale
production_repository_changed
production_runtime_changed
goal_model_fitted
prediction_data_used
```

Recommended boundary values:

```text
production_repository_changed = false
production_runtime_changed = false
goal_model_fitted = false
prediction_data_used = false
```

---

# 19. Implementation Boundary

The benchmark implementation should live under the research layer.

Suggested structure:

```text
research/studies/
└── study_089_aggregation_mathematics/
    ├── aggregation_axioms.md
    ├── synthetic_benchmark_specification.md
    ├── benchmark_scenarios.md
    ├── study_plan.md
    ├── run_synthetic_aggregation_benchmark.py
    └── outputs/
```

Reusable aggregation functions should not be buried inside the study
script.

If the mathematical functions are expected to support later empirical
studies, they should live in a research module such as:

```text
research/player_intelligence/
    aggregation_functions.py
```

The study runner should consume those functions.

---

# 20. Completion Criteria

The synthetic benchmark is complete when:

- every mandatory axiom has been tested;
- every candidate specification has a frozen parameter record;
- stability and responsiveness scenarios have been evaluated;
- rank-boundary behavior has been documented;
- depth behavior has been compared;
- failures have explicit diagnostics;
- no production artifact has changed;
- the report distinguishes mathematical failure from football
  hypothesis differences.

---

# 21. Study Boundary

This specification does not decide which aggregation is best.

It defines the experiments required to understand each method before
real football data are introduced.

The next document should define the exact scenario registry and expected
outcomes:

`benchmark_scenarios.md`