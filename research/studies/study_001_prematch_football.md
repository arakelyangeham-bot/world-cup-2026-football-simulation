# Study 001 — A Framework for Empirical Football Observation

### Establishing the Football Observatory Using Pre-Match Observations

---

# Abstract

This study establishes the methodological foundation for the Football Observatory, the empirical research framework developed as part of the World Cup 2026 Computational Football Project.

Rather than proposing a new predictive model, this study introduces a systematic methodology for measuring observable properties of football using historical international matches.

The Observatory is designed to compare generated football against historical football, quantify observable differences, and guide future match-generator research using empirical evidence instead of intuition.

Version 1 of the Observatory focuses exclusively on information available before kickoff together with the final match outcome. Event-level observations such as shots, possession, substitutions, and goal timing are intentionally excluded from this first phase.

The principal contribution of this study is therefore methodological. It demonstrates that football relationships can be measured using structured match observations, football observables, conditional relationships, uncertainty-aware estimation, and reusable statistical infrastructure.

---

# 1. Introduction

Most football prediction systems focus primarily on prediction.

Typical questions include:

- Who will win?
- What is the expected score?
- Who is most likely to win the tournament?

These questions are important.

However, they are not sufficient for building realistic football simulations.

A football simulator should not merely predict outcomes.

It should reproduce the observable statistical behaviour of football itself.

The purpose of the Football Observatory is therefore not prediction.

Its purpose is observation.

Instead of asking

> "Who wins?"

the Observatory asks

> "How does football behave?"

This distinction forms the philosophical foundation of the Computational Football phase of this project.

---

# 2. Motivation

Earlier phases of the project produced a complete football simulation platform consisting of:

- Team Repository
- Expected-goals model
- Match Generator
- Tournament Simulator
- Monte Carlo framework
- Production benchmarking framework

These components provide the ability to generate football matches.

The remaining challenge is determining whether the generated football behaves realistically.

This requires an instrument capable of measuring football itself.

The Football Observatory is that instrument.

---

# 3. Research Question

This first study investigates the following methodological question:

> Can observable football relationships be measured reliably using only pre-match information and final match outcomes?

The study deliberately avoids proposing new football models.

Instead it evaluates whether the Observatory provides a sufficiently robust empirical foundation for future research.

---

# 4. Data

Version 1 of the Observatory uses the historical international match dataset developed during previous project phases.

Each historical match contains:

## Pre-match observations

- Team-strength features
- FIFA rating points
- Poisson-derived strength estimates

## Final observations

- Final scoreline
- Match result

No event-level information is currently included.

The Observatory therefore studies relationships between conditions before kickoff and the observable outcome of the match.

---

# 5. Observatory Architecture

The Observatory converts historical football into structured observations.

```text
Historical Match
        │
        ▼
MatchObservation
        │
        ▼
Football Observable
        │
        ▼
Football Relationship
        │
        ▼
Relationship Analysis
        │
        ▼
Football Law (future)

Each layer has a single responsibility.

MatchObservation

Represents one historical football match.

FootballObservable

Represents a measurable football property.

Examples include:

Draw
One-goal match
Clean sheet
Both teams scored
High-scoring match
FootballRelationship

Represents the relationship between

one pre-match variable

and

one football observable.

Examples include:

FIFA points difference
        ↓
Draw rate

Attack difference
        ↓
Both teams scored

Goalkeeper difference
        ↓
Clean-sheet frequency

# 6. Statistical Methodology

Version 1 of the Observatory includes:

reusable relationship definitions
quantile-based binning
Wilson confidence intervals
reusable observable definitions

These choices are intended to maximize interpretability while minimizing methodological bias.

Quantile binning was adopted because it produces bins with approximately equal statistical power.

Wilson confidence intervals were selected because they provide robust interval estimates for binomial proportions, particularly for moderate sample sizes.

# 7. Initial Observatory Measurements

The Observatory currently measures several core football observables including:

Draw frequency
Home-win frequency
Away-win frequency
One-goal matches
Clean sheets
Both-teams-to-score
High-scoring matches
Blowouts
Common scorelines

Conditional response curves have been successfully produced using FIFA points difference as the first conditioning variable.

These measurements demonstrate that the Observatory can estimate football relationships together with statistical uncertainty.

# 8. Limitations

The Observatory currently measures only relationships between:

Pre-match state
        ↓
Final match outcome

It cannot yet study:

Possession
Match tempo
Goal timing
Tactical transitions
Substitutions
Game-state dynamics

These require event-level datasets and will form Phase 5B of the project.

# 9. Implications

The Football Observatory establishes a new methodology for Computational Football.

Future research should no longer begin by proposing new football models.

Instead, future work should follow the sequence:

Observation

↓

Relationship

↓

Hypothesis

↓

Model

↓

Benchmark

↓

Promotion

This ensures that every addition to the Football Match Generator is motivated by measurable football phenomena.

# 10. Future Work

Study 001 establishes the Observatory itself.

Subsequent studies will use this instrument to investigate specific football relationships.

Potential studies include:

Study 002 — One-Goal Football
Study 003 — Draw Dynamics
Study 004 — Clean Sheets
Study 005 — Competitive Balance
Study 006 — Home Advantage

Each study will investigate one football phenomenon using the common Observatory methodology introduced here.

What We Learned
Football can be represented as structured observations.
Football observables can be defined independently of prediction models.
Conditional football relationships can be measured from pre-match information.
Statistical uncertainty can be incorporated directly into Observatory measurements.
The Observatory is sufficiently mature to begin empirical football research.
What We Still Don't Know

This study deliberately leaves several questions unanswered.

We do not yet know:

how football observables evolve during matches,
how tempo changes throughout a match,
how game state alters scoring behaviour,
how substitutions affect match outcomes,
how tactical styles influence observable football.

Answering those questions will require future Observatory extensions using event-level football data.

Conclusion

Study 001 does not propose a new football model.

Instead, it establishes the Football Observatory as the empirical measurement framework for the Computational Football phase of the project.

The principal contribution of this work is methodological.

Future football models should be motivated by relationships discovered through the Observatory rather than by intuition alone.

In this sense, Study 001 marks the transition of the project from software engineering toward empirical computational football research.