experiment_003_recency_aware_representation.md

# Study 010

## Experiment 003

# Recency-Aware Player Representation

---

## Objective

Investigate whether more recent football evidence should contribute more strongly to Player Representation than older evidence.

---

## Motivation

Experiments 001 and 002 demonstrated that evidence quantity and evidence quality influence player representation.

However, all evidence is currently treated as temporally equivalent.

Football performance changes over time.

Player Representation should account for this.

---

## Research Question

Should performances from recent seasons contribute more strongly than performances from older seasons?

---

## Null Hypothesis

Historical performances should contribute equally regardless of age.

---

## Alternative Hypothesis

Recent football evidence should contribute more strongly than older evidence.

---

# Candidate Strategies

## Strategy A

Current production

Existing aggregation weights.

---

## Strategy B

Linear recency

Current season

↓

Previous season

↓

Older seasons

---

## Strategy C

Exponential decay

Recent seasons dominate progressively.

---

## Strategy D

Competition × Recency

Combine competition weighting with temporal weighting.

---

## Strategy E

Adaptive recency

Weight recent performances more strongly when evidence volume is high.

---

# Inputs

season_year

competition_manifest

recency_weight

total_weighted_evidence

minutes_played

---

# Measurements

Player rating variance

↓

Team representation variance

↓

Expected goals

↓

Scoreline realism

↓

Tournament realism

---

# Success Criteria

A recency-aware representation should:

- remain interpretable,
- preserve meaningful player separation,
- improve downstream realism,
- avoid excessive volatility.

---

# Non-Goals

No match engine modifications.

No Poisson model changes.

No goal sampler changes.

---

# Guiding Principle

Football ability is dynamic.

Player Representation should evolve with time.