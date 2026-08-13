benchmark_scenarios

# Study 089A — Benchmark Scenario Registry

## 1. Purpose

This document defines the canonical scenario registry used by the
synthetic aggregation benchmark.

Unlike `synthetic_benchmark_specification.md`, which describes the
benchmark framework, this document freezes the exact scenarios that
constitute Version 2B's first aggregation benchmark.

The scenarios should remain fixed for the duration of Version 2B.

Future scenarios should receive new identifiers rather than modifying
existing ones.

This preserves reproducibility across aggregation studies.

---

# 2. Scenario Naming Convention

Every scenario receives a stable identifier.

```
<Category>-<Number>
```

Categories are:

| Prefix | Purpose |
|----------|----------|
| AX | Mathematical axiom validation |
| ST | Stability |
| ER | Elite responsiveness |
| DS | Distribution shape |
| DP | Depth |
| RB | Rank boundary |
| SC | Scale consistency |
| SR | Structural role |
| RG | Regression |

Scenario IDs are permanent.

Once assigned, they should never be reused.

---

# 3. Mathematical Axiom Registry

## AX-001

### Name

Determinism

### Purpose

Verify repeated evaluation produces identical output.

### Population

```
0.91
0.88
0.84
0.80
0.76
0.71
0.66
```

### Transformation

None.

Evaluate repeatedly.

### Expected Result

Identical output.

---

## AX-002

### Name

Permutation Invariance

### Population

Same as AX-001.

### Transformations

- ascending
- descending
- fixed shuffle
- random permutation (seeded)

### Expected Result

No change.

---

## AX-003

### Name

Monotonic Improvement

### Population

```
0.90
0.85
0.80
0.75
0.70
```

### Transformation

Increase each player independently by

```
0.01
```

### Expected Result

Representation never decreases.

---

## AX-004

### Name

Continuity

### Population

```
0.90
0.85
0.80
0.75
0.70
```

### Transformations

Perturb one player by

```
±0.000001
±0.0001
±0.01
```

### Expected Result

Small perturbations produce proportionally small output changes.

---

## AX-005

### Name

Identity

### Populations

Uniform values:

```
0.00
0.25
0.50
0.75
1.00
```

### Expected Result

Aggregation equals the common value.

---

## AX-006

### Name

Boundedness

### Populations

- all zeros
- all ones
- mixed
- random

### Expected Result

Scalar strength remains within the player range.

---

# 4. Stability Registry

## ST-001

Weakest starter downgrade

Baseline

```
0.92
0.88
0.84
0.80
0.76
```

Modified

```
0.92
0.88
0.84
0.80
0.75
```

Purpose

Measure ordinary rotation sensitivity.

Expected

Very small change.

---

## ST-002

Elite starter downgrade

Baseline

```
0.92
0.88
0.84
0.80
0.76
```

Modified

```
0.91
0.88
0.84
0.80
0.76
```

Purpose

Compare elite sensitivity.

Expected

Still small.

Potentially larger than ST-001.

---

## ST-003

Near-identical fifth/sixth swap

Baseline

```
0.90
0.85
0.80
0.75
0.7001
0.7000
```

Modified

```
0.90
0.85
0.80
0.75
0.7000
0.7001
```

Purpose

Measure threshold robustness.

Expected

Near zero.

---

# 5. Elite Responsiveness Registry

## ER-001

Elite removal

Baseline

```
0.98
0.86
0.84
0.82
0.80
0.76
```

Modified

```
0.86
0.84
0.82
0.80
0.76
```

Purpose

Measure response to losing a superstar.

Expected

Clear reduction.

---

## ER-002

Ordinary starter removal

Baseline

Same as ER-001.

Modified

Remove

```
0.80
```

instead.

Purpose

Compare against ER-001.

Expected

Smaller reduction.

---

## ER-003

Superstar addition

Baseline

```
0.84
0.83
0.82
0.81
0.80
```

Modified

```
0.99
0.84
0.83
0.82
0.81
```

Purpose

Measure positive elite responsiveness.

Expected

Increase.

---

# 6. Distribution Registry

## DS-001

Balanced

```
0.85
0.85
0.85
0.85
0.85
```

versus

Top-heavy

```
0.95
0.90
0.85
0.80
0.75
```

Arithmetic means equal.

Purpose

Determine whether aggregation distinguishes distribution shape.

---

## DS-002

Extreme superstar

```
0.99
0.70
0.70
0.70
0.70
```

versus

Balanced

```
0.758
0.758
0.758
0.758
0.758
```

Purpose

Measure encoded superstar hypothesis.

---

## DS-003

Elite core

```
0.92
0.90
0.88
0.86
0.84
```

Balanced core

```
0.88
0.88
0.88
0.88
0.88
```

Purpose

Determine whether top-heavy cores are distinguishable.

---

# 7. Depth Registry

## DP-001

Roster expansion

Add:

```
0.30
0.25
0.20
0.15
0.10
```

Purpose

Measure fringe-player sensitivity.

---

## DP-002

Replacement improvement

Improve only

Ranks

```
6–10
```

Purpose

Measure replacement quality.

Expected

Primary strength unchanged.

Depth improved.

---

## DP-003

Uniform improvement

Improve every player by

```
0.02
```

Purpose

Verify scale consistency.

---

# 8. Rank Boundary Registry

## RB-001

Single threshold crossing

Swap fifth and sixth.

---

## RB-002

Three-way tie

```
0.70
0.70
0.70
```

Purpose

Tie handling.

---

## RB-003

Cluster around threshold

Generate:

```
0.7000

0.7001

0.6999

0.7002

0.6998
```

Purpose

Sensitivity audit.

---

# 9. Scale Registry

## SC-001

Uniform additive shift

Increase every player

```
+0.01
```

---

## SC-002

Uniform multiplicative increase

Multiply

```
×1.05
```

---

## SC-003

Equivalent scales

Represent identical populations on

```
[0,1]
```

and

```
[0,100]
```

Purpose

Normalization audit.

---

# 10. Structural Registry

These scenarios require richer player records.

They should be included after scalar aggregation validation.

---

## SR-001

Balanced XI

---

## SR-002

No defenders

---

## SR-003

No striker

---

## SR-004

Fallback-heavy XI

---

# 11. Regression Registry

Regression scenarios ensure that future implementations preserve
previously validated behavior.

Whenever an aggregation bug is discovered, a new regression scenario
should be added here.

Examples:

```
RG-001
```

Rank-boundary discontinuity.

```
RG-002
```

Softmax normalization error.

```
RG-003
```

Weight vector not summing to one.

Regression scenarios are cumulative.

They should never be removed.

---

# 12. Benchmark Coverage Matrix

| Property | Covered By |
|-----------|------------|
| Determinism | AX-001 |
| Permutation invariance | AX-002 |
| Monotonicity | AX-003 |
| Continuity | AX-004 |
| Identity | AX-005 |
| Boundedness | AX-006 |
| Stability | ST-001, ST-002, ST-003 |
| Elite responsiveness | ER-001, ER-002, ER-003 |
| Distribution awareness | DS-001, DS-002, DS-003 |
| Depth validity | DP-001, DP-002, DP-003 |
| Rank robustness | RB-001, RB-002, RB-003 |
| Scale consistency | SC-001, SC-002, SC-003 |
| Structural awareness | SR-001–SR-004 |
| Regression safety | RG-* |

---

# 13. Scenario Freeze Policy

Version 2B freezes this registry.

Future studies may:

- add new scenarios;
- add new categories;
- expand player records.

They should not:

- redefine existing populations;
- change expected outcomes;
- recycle identifiers.

This ensures that aggregation studies remain directly comparable over
time.

---

# 14. Study Completion

The synthetic benchmark is considered complete when every registered
scenario has been executed against every aggregation specification.

The benchmark should therefore be viewed as a fixed experimental
protocol rather than a collection of ad hoc examples.