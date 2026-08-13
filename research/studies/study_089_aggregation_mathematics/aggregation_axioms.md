aggregation_axioms.md

# Study 089A — Aggregation Axioms

## 1. Purpose

This document defines the mathematical properties expected of every
team-representation aggregation function considered during Version 2B.

Unlike Study 088A, which established design philosophy, this document
establishes testable mathematical requirements.

These axioms form the contract that every aggregation method should be
evaluated against before empirical benchmarking.

An aggregation function may intentionally violate a non-essential axiom,
but such violations must be documented and justified.

---

# 2. Definition

Let

```
P = {p₁, p₂, ..., pₙ}
```

represent a collection of player projections for one football
dimension.

An aggregation function

```
A(P)
```

maps the player population to a scalar team-strength feature.

Examples include:

- arithmetic mean;
- weighted mean;
- power mean;
- softmax-weighted mean;
- role-slot aggregation.

The purpose of this document is independent of any particular formula.

---

# 3. Core Axioms

## Axiom 1 — Determinism

For identical inputs,

```
A(P)
```

must always produce identical outputs.

No randomness should appear inside the aggregation layer.

Randomness belongs in downstream simulation.

---

## Axiom 2 — Permutation Invariance

Player ordering must not affect the representation.

If two player populations differ only by ordering,

```
P

↓

permute(P)
```

then

```
A(P)
=
A(permute(P))
```

This prevents accidental dependence on dataframe ordering or roster
serialization.

---

## Axiom 3 — Monotonicity

Increasing one player's projection while holding all other players
constant must not decrease team strength.

If

```
pᵢ' ≥ pᵢ
```

then

```
A(P')
≥
A(P)
```

This formalizes the intuitive idea that improving a player should never
make the team appear weaker.

---

## Axiom 4 — Continuity

Small player changes should produce small representation changes.

If player projections change by only a small amount,

the representation should also change by a small amount.

This protects the model from numerical instability and excessive
sensitivity to rating noise.

---

## Axiom 5 — Boundedness

The team representation should remain within the range implied by the
player projections.

For example,

if every player projection lies within

```
[0, 1]
```

then the aggregated representation should also lie within

```
[0, 1]
```

Aggregation should not create impossible football strength.

---

## Axiom 6 — Identity

If every player has identical projection

```
x
```

then

```
A(P)
=
x
```

The aggregation should preserve constant populations.

---

## Axiom 7 — Symmetry

Players with identical projections should contribute identically.

No player should receive special treatment based solely on an arbitrary
identifier.

---

# 4. Football Behavioral Properties

The following properties are not universal mathematical axioms.

They represent desired football behavior.

---

## Property 1 — Stability

Replacing one player with a nearly identical player should produce only
a small representation change.

Example:

```
82

↓

81
```

This protects against lineup uncertainty.

---

## Property 2 — Elite Responsiveness

Replacing an elite player with an average player should produce a
substantially larger change.

Example:

```
98

↓

72
```

The representation should recognize genuinely important football
losses.

---

## Property 3 — Depth Isolation

Adding weak fringe players should have little influence on first-team
strength.

Depth measures should change.

Primary strength should remain largely unchanged.

---

## Property 4 — Structural Awareness

Removing every natural defender should affect the representation in some
observable way.

The current representation intentionally satisfies this only weakly.

Future aggregations may strengthen this behavior.

---

## Property 5 — Distribution Awareness

Two teams with identical arithmetic means but different player
distributions should not necessarily receive identical representations.

Example:

```
95
90
85
80
75
```

versus

```
85
85
85
85
85
```

Whether the representation distinguishes them depends on the chosen
aggregation family.

---

# 5. Desirable Operational Properties

Every aggregation should also be evaluated on practical grounds.

## Interpretability

Can the output be explained in football language?

---

## Computational efficiency

Can the aggregation be computed quickly for thousands of simulated
matches?

---

## Reproducibility

Does identical input always reproduce identical output?

---

## Extensibility

Can new football information be added without redesigning the entire
aggregation?

---

## Modularity

Can the aggregation be replaced without changing downstream interfaces?

---

# 6. Categories of Requirements

Not every property has equal importance.

### Mandatory

- Determinism
- Permutation invariance
- Monotonicity
- Continuity
- Identity
- Boundedness
- Reproducibility

Violation requires explicit justification.

---

### Strongly Preferred

- Stability
- Interpretability
- Computational efficiency
- Modularity

---

### Research Hypotheses

These are not mandatory.

They are football questions.

Examples:

- elite responsiveness;
- distribution awareness;
- structural awareness;
- replacement sensitivity;
- nonlinear player influence.

Different aggregation families intentionally make different choices
here.

---

# 7. What This Document Does Not Decide

This document does not determine:

- the number of contributors;
- weighting schemes;
- role-slot definitions;
- depth formulas;
- nonlinear transformations.

Those remain experimental variables.

This document merely defines the evaluation contract.

---

# 8. Provisional Conclusion

Study 089A begins by separating mathematical correctness from football
hypotheses.

Every acceptable aggregation should satisfy a common mathematical
foundation.

Beyond that foundation, competing aggregation families are free to make
different football assumptions, provided those assumptions are
explicitly documented and empirically tested.

This distinction allows future representation research to compare
football ideas without repeatedly questioning the mathematical validity
of the aggregation itself.