study_022_league_competition_prototype.md

# Study 022 — League Competition Prototype

## Motivation

Competition Framework v1 successfully expressed knockout-style competitions through the invitational and domestic cup prototypes.

The next validation target is a league competition.

Unlike cups, leagues are standings-first competitions. They do not require brackets or knockout ties. This makes them an important second competition family for validating the framework.

## Research Question

Can the generic competition framework express a simple league competition without creating league-specific engine logic?

## Format

The prototype uses:

- 8 teams
- single round-robin
- 7 matches per team
- 28 total matches
- standard 3-1-0 points system
- champion determined by final standings

## Framework Flow

```text
Team list
    ↓
Round-robin match list
    ↓
Stage(type=LEAGUE)
    ↓
StageResolver
    ↓
StandingsEngine
    ↓
StageResult
    ↓
CompetitionEngine
    ↓
CompetitionResult.champion

Scope

This study does not model:

home/away balance
double round-robin schedules
matchdays
fixture congestion
player rotation
injuries
fatigue
promotion/relegation
real league data
Success Criteria

Study 022 succeeds if:

An 8-team league can be represented as a Competition.
The league can be represented as one Stage(type=LEAGUE).
The league table can be resolved by StandingsEngine.
The champion can be inferred from final standings.
No league-specific engine is created.
Strategic Importance

This validates that the framework supports both major football competition families:

knockout competitions
standings-based league competitions

Together with the domestic cup prototype, this demonstrates that the Competition Framework is no longer tied to tournament-style formats.