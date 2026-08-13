player_intelligence_production_evaluation_plan.md

# Player Intelligence Production Evaluation Plan

## Purpose

Evaluate whether the Player Intelligence pipeline is ready to become the default team representation for the World Cup 2026 simulator.

This document defines the production acceptance criteria.

---

# Current Candidates

Legacy Repository

Dimension-Specific Repository

Top-11 Repository

Star-Weighted Repository

---

# Selected Production Candidate

Dimension-Specific Repository

Reason:

Current benchmarks indicate the strongest combination of:

- scoreline realism
- tournament realism
- football interpretability
- architectural clarity

---

# Evaluation Pipeline

Stage 1

Repository Audit

Completed

---

Stage 2

Scoreline Benchmark

Completed

---

Stage 3

Tournament Benchmark

Completed

---

Stage 4

Football Observatory Comparison

Pending

---

Stage 5

Production Decision

Pending

---

# Acceptance Criteria

The Player Intelligence repository should:

✓ Match or improve scoreline TVD.

✓ Produce plausible tournament statistics.

✓ Produce realistic champion distributions.

✓ Preserve simulator stability.

✓ Remain fully interpretable.

---

# Production Integration Strategy

No production code should be removed.

The simulator should support:

```text
TEAM_REPOSITORY_SOURCE =
    legacy

TEAM_REPOSITORY_SOURCE =
    dimension_specific
```

through configuration.

---

# Rollback Strategy

If Player Intelligence fails future benchmarks:

Switch configuration back to:

```text
legacy
```

No code changes required.

---

# Versioning

Current Status

Player Intelligence Version 1.0

Status:

Production Candidate

---

# Future Improvements

Player Development

Evidence History

Dynamic Ability

Chemistry

Tactical Fit

Squad Rotation

Lineup Prediction

---

# Guiding Principle

Promote Player Intelligence to production only after evidence demonstrates that it consistently improves football realism.

Architecture alone is not sufficient.

Behavior determines production readiness.