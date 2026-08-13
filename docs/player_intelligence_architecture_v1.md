player_intelligence_architecture_v1.md

# Player Intelligence Architecture v1

## Purpose

Document the stable Player Intelligence architecture before production integration.

This version defines the object pipeline that transforms raw player evidence into team-level simulation inputs.

---

## Architecture

```text
PlayerEvidenceRepository
        ↓
PlayerRepository
        ↓
PlayerRepresentationEngine
        ↓
RosterBuilder
        ↓
StartingXIBuilder
        ↓
TeamRepresentationBuilder
        ↓
TeamRepositoryBuilder

Layer Responsibilities
PlayerEvidenceRepository

Owns raw player evidence history.

Input:

data/raw/sofascore/sofascore_player_stats.csv

Output:

PlayerEvidenceHistory

Must not:

compute final player ability
build squads
build team strength
PlayerRepository

Owns canonical Player objects.

Input:

data/processed/player_ratings.csv

Optional input:

PlayerEvidenceRepository

Output:

Player

Must not:

choose lineups
compute team strength
simulate matches
PlayerRepresentationEngine

Owns interpretation of player knowledge.

Input:

Player
RoleRatings
PlayerEvidence
PlayerEvidenceHistory

Output:

PlayerRepresentation

Current v1 representation includes:

current ability
evidence confidence
total minutes
competition count
season count
latest season
recency share

Must not:

build squads
build team repositories
simulate matches
RosterBuilder

Owns squad construction.

Input:

PlayerRepository

Output:

Squad

Must not:

rate players
choose starting XI
simulate matches
StartingXIBuilder

Owns expected lineup construction.

Input:

Squad or player ratings table
formation manifest

Output:

StartingXI
expected_lineups.csv

Must not:

compute player ratings
compute match scores
TeamRepresentationBuilder

Owns transformation from players or lineups into team representation.

Input:

Squad
StartingXI
RoleProjection

Output:

TeamRepresentation

Must not:

know about SofaScore
know about CSV schemas
simulate matches
TeamRepositoryBuilder

Owns projection from football representation into simulation inputs.

Input:

TeamRepresentation

Output:

TeamRepositoryEntry

Must not:

compute player ratings
choose lineups
simulate matches
Stable Design Principle

Player Intelligence should transform football knowledge into simulation-ready representations.

It should not contain tournament logic, goal-sampling logic, or match-engine logic.

Current Status

Completed:

Player schema
Role ratings
Evidence summary
Evidence history
Player representation
Roster builder
Starting XI builder
Team representation builder
Team repository builder
Smoke tests
Production Integration Target

The next phase is to connect:

Player Intelligence v1
        ↓
production team repository
        ↓
scoreline-first simulator

without changing the match engine.

Open Research Questions
What scale should current ability use?
Should ability be z-score, percentile, 0–10, or Elo-like?
Should team representation use full squad or expected XI?
How should evidence history influence current ability?
How should recency and player development be modeled?
Guiding Principle

Do not tune the simulator to compensate for weak representations.

Improve the football representation first.