version_2b_representation_roadmap

# Study 088A — Version 2B Representation Roadmap

## 1. Purpose

This document defines the research roadmap for the next generation of
team representations.

It translates the design principles established in Study 088A into an
ordered implementation strategy.

The roadmap intentionally emphasizes incremental scientific validation
over rapid architectural change.

The objective is not to discover the most sophisticated representation.

The objective is to identify the smallest representation improvement
that produces measurable football value.

---

# 2. Current Evidence

Studies completed before this roadmap have produced several important
findings.

## Study 085A

Bundesliga replay identified systematic prediction bias.

---

## Study 085B

Calibration improved predictive performance only modestly.

Representation quality remained the dominant limitation.

---

## Study 086

Architecture audit demonstrated that the production framework already
supports interchangeable team representations.

No architectural redesign is required.

---

## Study 087A

Reliable season-level player usage features were successfully
constructed.

---

## Study 087B

Usage-informed player selection altered very few projected starting
lineups.

---

## Study 087C

Those lineup changes produced only small changes in aggregated team
representations.

The current aggregation is intentionally stable and resistant to minor
player substitutions.

---

## Conclusion

The remaining opportunity is unlikely to come from further refinement
of lineup selection.

The next research phase should instead investigate how selected players
are aggregated into team-level football intelligence.

---

# 3. Research Philosophy

Version 2B will follow four principles.

## Principle 1

One hypothesis at a time.

Each study should isolate one representation idea.

---

## Principle 2

Augment before replacing.

The validated baseline should remain available throughout the research
process.

---

## Principle 3

Interpretability first.

Simple football concepts should be exhausted before introducing complex
mathematical models.

---

## Principle 4

Production follows evidence.

No representation enters production merely because it appears more
realistic.

Promotion requires empirical validation.

---

# 4. Phase I — Low-Risk Augmentation

Goal:

Determine whether small additions improve the existing representation.

Candidate studies:

## Study 089A

Rank-weighted top-five aggregation.

Questions:

Does weighting elite contributors improve representation quality?

---

## Study 089B

Dimension-specific contributor counts.

Questions:

Should attack, midfield, and defense use different values of k?

---

## Study 089C

Replacement-drop-off depth.

Questions:

Does replacement quality describe squad strength better than whole-squad
means?

---

## Study 089D

Distribution-shape features.

Examples:

- top-five variance;
- concentration;
- star gap;
- quality spread.

Question:

Do two squads with identical means behave differently?

---

## Exit criterion

At least one augmentation demonstrates meaningful incremental
information while preserving stability and interpretability.

---

# 5. Phase II — Structural Information

Goal:

Determine whether tactical structure adds information beyond player
quality.

Candidate studies:

## Study 090A

Role-balance features.

Examples:

- natural striker coverage;
- center-back balance;
- defensive midfield coverage.

---

## Study 090B

Fallback-role penalties.

Questions:

Should structural compromises reduce representation quality?

---

## Study 090C

Formation robustness.

Questions:

How stable are representations across reasonable tactical variations?

---

## Exit criterion

Structural features explain variation not captured by dimensional
strength alone.

---

# 6. Phase III — Dynamic Representations

Goal:

Move from season-level player populations toward prediction-date-valid
representations.

Candidate studies:

## Study 091A

Availability estimation.

---

## Study 091B

Expected starter probabilities.

---

## Study 091C

Availability-weighted aggregation.

---

## Study 091D

Prediction-date repository construction.

---

## Exit criterion

Representations become valid for a specific match date without changing
the aggregation architecture.

---

# 7. Phase IV — Interaction Research

Goal:

Investigate football interactions that violate additive assumptions.

Potential topics:

- striker-creator interaction;
- center-back partnerships;
- winger-fullback overlap;
- midfield balance;
- goalkeeper-defense interaction.

These studies should remain exploratory until strong evidence justifies
their inclusion.

---

# 8. Phase V — Learned Aggregation

Only after interpretable representations have been thoroughly explored
should learned aggregation methods be considered.

Possible directions:

- Deep Sets;
- attention pooling;
- graph neural networks;
- learned permutation-invariant representations.

These approaches should supplement—not replace—the interpretable
baseline until substantial evidence exists.

---

# 9. Benchmarking Pipeline

Every candidate representation should pass through the same evaluation
sequence.

```
Design

↓

Mathematical validation

↓

Synthetic football scenarios

↓

Bundesliga representation audit

↓

Sensitivity analysis

↓

Feature comparison

↓

Goal-model benchmark

↓

Held-out validation

↓

Production consideration
```

No stage should be skipped.

---

# 10. Promotion Requirements

A representation should satisfy all of the following before production
promotion.

## Football

The representation captures a meaningful football concept.

---

## Mathematical

The aggregation behaves predictably under controlled scenarios.

---

## Empirical

The representation adds measurable information.

---

## Predictive

Held-out predictive performance improves.

---

## Architectural

Existing interfaces remain stable.

---

## Operational

The computational cost remains practical.

---

# 11. Long-Term Vision

The long-term architecture can be viewed conceptually as:

```
Player Intelligence
        ↓
Player Availability
        ↓
Expected Player Population
        ↓
Representation Aggregation
        ↓
Team Representation
        ↓
Observation Builder
        ↓
Goal Model
        ↓
Match Engine
        ↓
Tournament Simulator
```

Notice that representation aggregation remains one modular component.

Future improvements should occur inside this boundary without requiring
changes throughout the rest of the project.

---

# 12. What Version 2B Is Really About

Version 2A primarily improved architecture.

Version 2B shifts attention toward football intelligence.

The guiding question is no longer:

"How should the simulator be organized?"

It becomes:

"How should football knowledge be represented?"

This distinction marks the transition from software engineering to
football modeling.

---

# 13. Success Criteria

Version 2B will be considered successful if it produces:

- richer football representations;
- improved scientific understanding;
- reproducible benchmarks;
- modest but genuine predictive improvements;
- reusable architecture for future research.

Success is **not** defined by the number of new features added.

It is defined by increasing explanatory power while preserving
interpretability.

---

# 14. Final Remarks

Study 088A intentionally concludes without recommending one definitive
aggregation method.

Instead, it establishes a disciplined framework for evaluating future
ideas.

The current representation has demonstrated valuable properties:

- stability;
- interpretability;
- portability;
- modularity.

Future work should preserve these strengths while carefully introducing
greater responsiveness, structural awareness, and football realism.

The guiding philosophy of Version 2B is therefore:

> Improve the representation by understanding football more deeply, not
> by making the mathematics unnecessarily more complicated.