research_protocol.md

# Research Protocol 009

# Team Representation

### A Study of Player-to-Team Representation for Football Simulation

---

# Protocol Status

Draft

Purpose:

Investigate how player-level information should be represented at the team level before entering the expected-goals model.

---

# Motivation

Recent calibration work demonstrated:

- The scoreline-first architecture is functioning correctly.
- The Poisson goal model is statistically well calibrated.
- The remaining discrepancies appear to originate in the representation of team strength.

This study investigates the mathematical representation of football teams rather than the calibration of the simulator.

---

# Primary Research Question

> What mathematical representation of a football team best preserves football strength?

---

# Secondary Research Questions

## RQ1

How much information is lost when players are collapsed into a single team-strength vector?

---

## RQ2

Do weighted averages adequately distinguish elite teams from average teams?

---

## RQ3

Should team representation be based upon:

- full squad,
- expected starting XI,
- or a hybrid?

---

## RQ4

Should different positional groups contribute independently to team representation?

---

## RQ5

How much squad depth should influence team strength?

---

# Candidate Representations

Current weighted averages

Expected Starting XI

Position-group composites

Depth-aware representations

Distribution summaries

Future latent representations

---

# Inputs

Player Repository

Expected Lineups

Player Ratings

Team Repository

Calibration audits

---

# Planned Analyses

Analysis 1

Current representation audit

---

Analysis 2

Representation dynamic range

---

Analysis 3

Position-group contribution

---

Analysis 4

Squad-depth sensitivity

---

Analysis 5

Expected-lineup comparison

---

# Success Criteria

The study succeeds if it identifies a representation that:

- preserves meaningful separation between teams,
- remains interpretable,
- is compatible with the Team Repository,
- improves expected-goals inputs.

---

# Non-Goals

No changes to:

- match engine,
- goal sampler,
- tournament simulator,
- Monte Carlo framework.

---

# Deliverables

Representation comparison

Representation recommendations

Updated Team Repository proposal

Player Intelligence recommendations

---

# Guiding Principle

Improve the representation before improving the simulator.

The quality of the simulator is fundamentally limited by the quality of the information it receives.