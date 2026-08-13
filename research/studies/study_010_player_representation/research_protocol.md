research_protocol.md

# Research Protocol 010

# Player Representation

### A Study of Mathematical Representations of Individual Football Ability

---

# Protocol Status

Draft

---

# Purpose

Investigate how individual footballers should be represented before being
aggregated into team-level football representations.

This study shifts the focus from team representation to the representation
of player ability itself.

---

# Motivation

Study 009 demonstrated that modifying role-projection weights produces only
modest changes to team representation.

This suggests that the primary bottleneck lies upstream in the Player object.

Therefore the next research question becomes:

> What information should a Player contain?

---

# Primary Research Question

How should football ability be represented mathematically at the player level?

---

# Secondary Questions

## RQ1

How many role-specific ratings should a player possess?

---

## RQ2

Should player ability be represented only by ratings?

Or should additional football concepts be represented?

---

## RQ3

How should evidence quality influence player representation?

---

## RQ4

How should multiple competitions contribute?

---

## RQ5

How should recent performances be weighted?

---

## RQ6

Should players possess uncertainty estimates?

---

# Current Player

```text
Identity

Availability

Overall Rating

Role Ratings
```

---

# Candidate Extensions

Evidence confidence

Sample quality

Competition history

Recent form

Role flexibility

Positional versatility

Availability

Club form

International form

Minutes

Age

Fatigue

Injury status

---

# Planned Experiments

Experiment 001

Evidence-aware player ratings

---

Experiment 002

Recency weighting

---

Experiment 003

Competition weighting

---

Experiment 004

Role specialization

---

Experiment 005

Player uncertainty

---

# Success Criteria

A richer Player representation should:

- produce more informative role ratings

- improve team representation

- improve expected goals

- improve scoreline realism

without changing the simulator architecture.

---

# Non-Goals

No changes to:

Match Engine

Goal Sampler

Tournament Simulator

Monte Carlo framework

---

# Deliverables

Updated Player object

Updated Player Repository

Improved Team Representation

Production evaluation

---

# Guiding Principle

Improve football knowledge before improving football simulation.