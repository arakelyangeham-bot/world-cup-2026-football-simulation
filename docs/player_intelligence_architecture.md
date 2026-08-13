player_intelligence_architecture.md

# Player Intelligence Architecture

## Purpose

Define how player-level information will eventually flow into the World Cup 2026 simulation platform without redesigning the existing match engine, tournament engine, or Observatory.

## Core Principle

The match engine should consume team-strength objects.

It should not care whether those strengths come from static ratings, FIFA points, player aggregation, expected lineups, or future models.

## Target Architecture

```text
Player Data
    ↓
Player Repository
    ↓
Squad Model
    ↓
Expected Starting XI
    ↓
Player-Derived Team Strength
    ↓
Expected Goals
    ↓
Scoreline Realization
    ↓
Match Engine
    ↓
Tournament Simulator

Key Objects
Player

Represents an individual footballer.

Possible fields:

player_id
name
national_team
club
primary_position
secondary_positions
rating
attack_rating
midfield_rating
defense_rating
goalkeeper_rating
recent_form
minutes_played
availability_status
Squad

Represents the available player pool for a national team.

Starting XI

Represents the expected selected players for a specific match.

Player-Derived Team Strength

Aggregates players into the same strength dimensions already used by the simulator:

attack
midfield
defense
gk
poisson_attack
poisson_defense
fifa_points
Migration Strategy
Stage 1

Keep current static team-strength tables.

Stage 2

Create player repository.

Stage 3

Create squad model.

Stage 4

Create expected lineup model.

Stage 5

Aggregate player data into team-strength objects.

Stage 6

Benchmark player-derived team strength against current static team strength.

Stage 7

Only replace production inputs if benchmarks improve.

What Changes Immediately?

Nothing.

No production code changes.

What This Enables Later
injuries
suspensions
expected lineups
player form
player aging
fatigue
tactical shape
dynamic national-team strength
Open Questions
Which player data source will be used?
How will expected lineups be estimated?
How will injuries be collected?
How will club form translate to national-team strength?
How will missing players be handled?
Guiding Principle

Player Intelligence should enrich the upstream team-strength layer without destabilizing the existing production simulator.


This gives us the blueprint without committing to a data source or implementation yet.