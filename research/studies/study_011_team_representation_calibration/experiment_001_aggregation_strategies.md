experiment_001_aggregation_strategies.md

# Study 011

## Experiment 001

# Aggregation Strategies

---

## Objective

Compare multiple football philosophies for aggregating individual player
representations into team representations.

The objective is not merely to average ratings, but to determine which
aggregation strategy best preserves football strength.

---

# Research Question

How should eleven footballers become one football team?

---

# Motivation

Player Intelligence Version 1 now provides:

- Player Representation
- Role Ratings
- Evidence
- Evidence History

The remaining modeling question concerns aggregation.

---

# Strategy A

## Uniform Squad Mean

All players contribute equally.

Advantages

Simple

Stable

Interpretable

Limitations

Ignores starting XI.

---

# Strategy B

## Top-N Mean

Current implementation.

Highest-rated players dominate.

Advantages

Simple

Reasonably football-like

Limitations

Bench ignored.

---

# Strategy C

## Expected Starting XI

Only projected starters contribute.

Advantages

Closest to real football.

Limitations

Ignores squad depth.

---

# Strategy D

## Starter + Bench

Expected XI

+

Reduced contribution from substitutes.

Advantages

Captures squad depth.

---

# Strategy E

## Star Player Weighting

Elite players receive nonlinear weighting.

Advantages

Captures teams built around stars.

Limitations

May exaggerate elite players.

---

# Strategy F

## Position-Balanced

Maintain explicit positional balance.

Advantages

Preserves football structure.

---

# Measurements

Dynamic range

↓

Variance

↓

Legacy overlap

↓

Expected goals

↓

Tournament realism

---

# Success Criteria

Preferred strategies should:

- preserve football realism
- increase meaningful team separation
- remain interpretable
- outperform the current representation

---

# Non-Goals

No simulator changes.

No Poisson changes.

No goal sampler changes.

---

# Guiding Principle

Aggregation is not averaging.

Aggregation is football modeling.