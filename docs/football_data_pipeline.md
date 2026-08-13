football_data_pipeline.md

# Football Data Pipeline

## Purpose

Define the canonical data-ingestion architecture for all future football data.

The objective is to automate player and team data acquisition while preserving the modular architecture of the World Cup 2026 project.

The pipeline must support both international and club football without requiring downstream architectural changes.

---

# Design Principles

The pipeline follows the same architectural principles as the rest of the project.

- Modular
- Replaceable
- Testable
- Reproducible
- Source-agnostic

No downstream component should depend upon a particular data provider.

---

# High-Level Architecture

```text
Competition
      ↓
Competition Adapter
      ↓
Team Discovery
      ↓
Roster Builder
      ↓
Player Repository
      ↓
Statistics Pipeline
      ↓
Feature Engineering
      ↓
Team Strength Aggregation
      ↓
Team Repository
      ↓
Expected Goals
      ↓
Scoreline Realization
      ↓
Tournament Simulation
```

---

# Layer 1 — Competition Discovery

Purpose:

Determine which teams participate.

Examples:

- FIFA World Cup
- UEFA Champions League
- Premier League
- MLS

Output:

```text
Competition
Teams
Schedule
```

No player information yet.

---

# Layer 2 — Team Discovery

Purpose:

Identify all participating teams.

Output:

```text
Team identifiers
```

No roster construction yet.

---

# Layer 3 — Roster Builder

Purpose:

Construct squad lists automatically.

Output:

```text
Team

↓

Players
```

Important:

The roster builder replaces the manually constructed roster currently used by the project.

This becomes the first fully automated stage.

---

# Layer 4 — Player Repository

Purpose:

Maintain canonical Player objects.

Input:

Any supported data source.

Output:

```text
Player

Squad

Starting XI
```

No statistics are computed here.

The repository stores football entities.

---

# Layer 5 — Statistics Pipeline

Purpose:

Collect player statistics.

Possible sources:

- SofaScore
- FBref
- Opta
- Understat
- Future providers

Statistics remain provider-specific until normalized.

---

# Layer 6 — Feature Engineering

Purpose:

Transform provider statistics into canonical football features.

Output:

```text
wc_model_features
```

Equivalent future outputs should exist for every supported competition.

---

# Layer 7 — Team Strength Aggregation

Purpose:

Aggregate players into team-level football strength.

Output:

```text
attack

midfield

defense

goalkeeper

poisson_attack

poisson_defense
```

Exactly matches the existing Team Repository schema.

---

# Layer 8 — Team Repository

Purpose:

Provide canonical team-strength objects to the simulator.

This interface should remain unchanged regardless of upstream data sources.

---

# Architectural Boundary

Everything below Team Repository already exists.

The Football Data Pipeline should enrich upstream information without requiring downstream redesign.

---

# Migration Strategy

Stage 1

Current manually constructed World Cup roster.

Stage 2

Automated roster construction.

Stage 3

Automated player statistics.

Stage 4

Canonical Player Repository.

Stage 5

Dynamic Starting XI prediction.

Stage 6

Player-derived Team Repository.

Stage 7

Retire manually maintained strength tables.

---

# Supported Competitions

The architecture should eventually support:

- FIFA World Cup
- Continental championships
- Domestic leagues
- Continental club competitions
- Friendly matches

No downstream architectural changes should be required.

---

# Guiding Principle

The simulator should never know where football data originated.

It should consume only canonical football domain objects.

Competition-specific scraping, provider-specific schemas, and roster construction belong entirely within the Football Data Pipeline.