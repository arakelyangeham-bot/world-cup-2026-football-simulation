PROJECT

World Cup 2026 Football Intelligence Platform
Guiding Philosophy

The project has one primary goal:

Learn as much as possible about football from historical data before simulating or predicting matches.

Simulation is the final consumer.

Prediction is the intermediate consumer.

Player intelligence is the foundation.

Everything flows upward from data.

Layer 0 — Competition Infrastructure

Everything begins with competitions and seasons.

Competition Registry
        ↓
Season Discovery
        ↓
Season Registry
        ↓
Competition Manifest
        ↓
Feature Manifest
Competition Registry

Static knowledge.

Examples:

Premier League
La Liga
Bundesliga
Serie A
Ligue 1

Champions League
Europa League

World Cup
Euro
Copa América
AFCON
Asian Cup

Contains:

competition identity
competition type
Sofascore unique tournament ID
metadata
Season Registry

Automatically discovers:

Competition

↓

Available seasons

↓

season_id

This is exactly what discover_sofascore_competition_seasons.py already does.

Competition Manifest

Operational configuration.

Determines:

enabled competitions
scrape priority
scrape stages
weighting

It orchestrates ingestion rather than describing football entities.

Feature Manifest

Describes data quality.

Not:

Should we scrape?

Instead:

Which statistical features are sufficiently available after scraping?

Layer 1 — Generic Ingestion

Once competitions and seasons are known:

Competition
        ↓
Season
        ↓
Teams
        ↓
Players
        ↓
Profiles
        ↓
Player-season statistics

The order matters.

Teams

Discover participating clubs or national teams.

Players

Discover every player participating in that competition-season.

This is where historical player-team membership originates.

Profiles

Player identity.

Positions.

Preferred foot.

Birth date.

Current descriptive information.

Profiles are not the source of historical club membership.

Statistics

Player-season evidence.

Minutes.

Ratings.

Appearances.

Performance.

Everything remains attached to:

competition
season
team
player
Layer 2 — Player Intelligence

This layer should remain almost entirely shared between club and international football.

Profiles
+
Player-season statistics
        ↓
Player Registry
        ↓
Player Evidence Repository
        ↓
Attribute Scores
        ↓
Role Ratings
        ↓
Player Representation

This is one of the strongest pieces of the current architecture.

Nothing here fundamentally cares whether the player represents:

France

or

Liverpool
Layer 3 — Team Intelligence

This is where the project begins to branch.

National Football
National roster builder
        ↓
Expected XI
        ↓
National Team Representation
        ↓
National Repository
Club Football
Club roster builder
        ↓
Expected XI
        ↓
Club Team Representation
        ↓
Club Repository

The important point:

Everything before the roster builder is shared.

Everything after the team representation is mostly shared.

The roster builder is the first major divergence.

Layer 4 — Prediction

Again, largely shared.

Repository
        ↓
Feature Builder
        ↓
Feature Sets
        ↓
Expected Goals
        ↓
Lambda

The current architecture already contains most of this:

feature builder,
feature sets,
calibrated lambda model,
heuristic lambda model.

Future additions belong here:

environment features,
lineup confidence,
club priors,
experimental features.
Layer 5 — Match Engine

Independent of how lambdas were produced.

Lambda

↓

Goal sampler

↓

Scoreline

Whether the lambdas came from:

heuristic equations,
Poisson GLM,
neural network,

the scoreline engine does not care.

That separation is one of the strongest design decisions in the project.

Layer 6 — Simulation

This is the final consumer.

National branch:

World Cup

Club branch:

Domestic League

Champions League

Europa League

Conference League

Club World Cup

The match engine is shared.

Competition rules differ.

Shared vs Branch-specific Components
Shared
Competition infrastructure

Ingestion

Player registry

Player evidence

Attribute scoring

Role ratings

Player representation

Feature builder

Expected goals

Match engine

Goal samplers

Benchmarking

Validation
National-specific
National roster builder

National repository

FIFA-based rating prior

World Cup simulator
Club-specific
Club roster builder

Transfer-aware memberships

Club repository

Opta / Club Elo / learned rating prior

League simulator

Champions League simulator
Research Layer

This is orthogonal to everything else.

Studies like:

PCA football environments

Aggregation strategies

Calibration

Expected lineups

Depth experiments

Feature ablations

do not belong to either branch.

They improve shared intelligence.

Validation Layer

Every major layer has dedicated validation.

Competition validation

↓

Ingestion validation

↓

Player validation

↓

Representation validation

↓

Repository validation

↓

Prediction validation

↓

Simulation validation

No architectural change reaches production without an equivalence test.

Long-Term Vision

This is the part I'm most excited about.

Eventually, the architecture should look like this:

Competition Infrastructure
        │
        ▼
Generic Football Ingestion
        │
        ▼
Player Intelligence
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
National Football              Club Football
        │                              │
        ▼                              ▼
National Repository            Club Repository
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Shared Prediction Pipeline
                       ▼
              Shared Match Engine
                       ▼
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼
World Cup Simulator         League / Champions League /
                            Club World Cup Simulators