player_intelligence_integration_plan.md

# Player Intelligence Integration Plan

## Purpose

Integrate Player Intelligence v1 into the production simulation pipeline without replacing the existing production path prematurely.

---

## Current Production Path

```text
wc_2026_model_features.csv
        ↓
sofascore_team_aggregator.py
        ↓
wc_2026_team_strength.csv
        ↓
team_strength_loader.py
        ↓
scoreline-first simulator

Proposed Parallel Path
player_ratings.csv
        ↓
PlayerEvidenceRepository
        ↓
PlayerRepository
        ↓
RosterBuilder
        ↓
StartingXIBuilder
        ↓
TeamRepresentationBuilder
        ↓
TeamRepositoryBuilder
        ↓
player_intelligence_team_repository.csv
Integration Rule

Do not replace production until the Player Intelligence path can produce a comparable team repository schema.

Required Output Schema

The Player Intelligence repository must eventually provide:

nation
att_composite
mid_composite
def_composite
gk_composite
poisson_attack_adj
poisson_defense_adj

or a compatible canonical replacement.

Phase 1 — Parallel Build

Create a new script:

scripts/build_player_intelligence_team_repository.py

Output:

outputs/player_intelligence/player_intelligence_team_repository.csv

This script should not overwrite production files.

Phase 2 — Comparison Audit

Compare:

data/processed/wc_2026_team_strength.csv

against:

outputs/player_intelligence/player_intelligence_team_repository.csv

Metrics:

row coverage
team overlap
attack distribution
midfield distribution
defense distribution
goalkeeper distribution
poisson attack distribution
poisson defense distribution
Phase 3 — Simulation Sandbox

Only after Phase 2 passes, allow the simulator to optionally load:

player_intelligence_team_repository.csv

behind a config flag.

Non-Goals

This integration will not:

change the match engine
change the goal sampler
overwrite production team-strength files
change tournament logic
Guiding Principle

Build the Player Intelligence production path in parallel first.

Replace production only after evidence shows the new representation is stable and useful.