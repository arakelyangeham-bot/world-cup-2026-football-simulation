experiment_001_evidence_representation.md

# Study 010

## Experiment 001

# Evidence-Aware Player Representation

---

## Objective

Investigate whether incorporating evidence quality into Player Representation
produces richer and more reliable team representations.

---

## Motivation

The current Player object stores:

- role ratings
- evidence confidence
- sample quality
- competition count
- minutes played

However, only role ratings currently contribute to Team Representation.

This experiment investigates whether evidence should influence player influence.

---

## Research Question

Should two players with identical role ratings contribute equally when their
available evidence differs substantially?

---

## Null Hypothesis

Role ratings alone are sufficient.

Evidence quality does not improve representation.

---

## Alternative Hypothesis

Evidence quality should influence player contribution.

Players supported by more evidence should contribute more confidently.

---

# Candidate Representations

## Representation A

Role Ratings only

(Current production)

---

## Representation B

Role Ratings

×

Evidence Confidence

---

## Representation C

Role Ratings

×

Minutes Weight

---

## Representation D

Role Ratings

×

Sample Quality

---

## Representation E

Combined Evidence Score

Role Rating

×

Evidence Confidence

×

Sample Quality

---

# Measurements

For each representation measure:

Player variance

↓

Team representation variance

↓

Expected-goals variance

↓

Scoreline realism

↓

Tournament realism

---

# Inputs

Player Repository

Role Ratings

Evidence Confidence

Minutes Played

Sample Quality

---

# Outputs

Updated Player Representation

Updated Team Representation

Comparison tables

---

# Success Criteria

Evidence-aware representations should:

- preserve football realism
- increase meaningful separation
- remain interpretable
- improve downstream calibration

without increasing architectural complexity.

---

# Non-Goals

No simulator changes.

No goal-sampler changes.

No tournament changes.

---

# Future Work

Experiment 002

Recency-aware representation

Experiment 003

Competition-aware representation

Experiment 004

Uncertainty-aware representation

---

# Guiding Principle

A player's influence should depend not only upon ability,
but also upon the quality of the evidence supporting that ability.