player_information_taxonomy.md

# Player Information Taxonomy

## Study

Study 010 — Player Representation

---

# Purpose

Define the canonical categories of information that describe a football player.

This taxonomy separates:

- football identity,
- observed evidence,
- inferred ability,
- contextual information,
- uncertainty.

The taxonomy serves as the conceptual foundation for all future Player Intelligence work.

---

# Design Principles

A Player object should represent football knowledge rather than data-source fields.

Every attribute should answer one football question.

No attribute should exist solely because a provider exposes it.

---

# Category A — Identity

Purpose:

Identify the footballer.

Examples:

- canonical_player_id
- player_name
- date_of_birth
- nationality
- dominant_foot

Properties:

Stable.

Rarely changes.

---

# Category B — Positional Information

Purpose:

Describe where the footballer can play.

Examples:

- primary_position
- secondary_positions
- eligible_roles
- positional_flexibility

Properties:

Changes slowly.

---

# Category C — Ability

Purpose:

Represent football quality.

Examples:

- role_ratings
- attacking ability
- defensive ability
- passing
- finishing
- aerial ability
- shot stopping

Properties:

Changes gradually.

Learned from football evidence.

---

# Category D — Evidence

Purpose:

Describe why we believe the ability estimates.

Examples:

- minutes_played
- competitions_observed
- season_count
- evidence_confidence
- sample_quality
- recency_weight

Properties:

Supports interpretation.

Does not directly represent football ability.

---

# Category E — Availability

Purpose:

Represent whether the footballer can play.

Examples:

- available
- injured
- suspended
- expected_to_start
- fitness

Properties:

Highly dynamic.

---

# Category F — Form

Purpose:

Represent short-term performance.

Examples:

- recent_form
- club_form
- national_team_form
- momentum

Properties:

Changes rapidly.

---

# Category G — Context

Purpose:

Represent the football environment.

Examples:

- current_club
- competition
- league
- manager
- tactical_system

Properties:

External to player ability.

---

# Category H — Uncertainty

Purpose:

Represent confidence in the player model.

Examples:

- rating_uncertainty
- role_uncertainty
- evidence_uncertainty

Properties:

Derived.

Should accompany predictions.

---

# Information Flow

```text
Football Evidence
        ↓
Player Information
        ↓
Player Representation
        ↓
Role Ratings
        ↓
Team Representation
        ↓
Expected Goals
        ↓
Simulation
```

---

# Guiding Principle

Player Representation should describe football knowledge, not data-provider schemas.

The Player object should remain valid regardless of whether the information originated from:

- SofaScore
- FBref
- Opta
- StatsBomb
- Manual scouting
- Future providers

Only adapters should depend upon provider-specific fields.

---

# Future Experiments

Experiment 001

Evidence-aware Player Representation

Experiment 002

Recency-aware Player Representation

Experiment 003

Competition-aware Player Representation

Experiment 004

Uncertainty-aware Player Representation

Experiment 005

Dynamic Player Representation

---

# Long-Term Vision

The Player object becomes the canonical unit of football knowledge.

Everything downstream derives from it.