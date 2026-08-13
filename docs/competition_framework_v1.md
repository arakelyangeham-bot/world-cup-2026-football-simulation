# Competition Framework v1

## Motivation

The project began with a World Cup 2026-specific simulator. As the project evolved, it became useful to separate football competition structure from any one tournament format.

Competition Framework v1 introduces reusable abstractions for defining, resolving, and analyzing football competitions beyond the 2026 World Cup.

## Core Architecture

The framework follows this pipeline:

```text
Competition
    ↓
Stage
    ↓
StageResolver
    ↓
Stage Engine
    ↓
StageResult
    ↓
AdvancementRule
    ↓
AdvancementResult
    ↓
CompetitionResult

Core Models
Competition
CompetitionResult
Stage
StageResult
MatchResult
Tie
TieResult
StandingRow
StandingsTable
Engines
StandingsEngine
KnockoutEngine
StageResolver
CompetitionEngine
Rules
AdvancementRule
TopNAdvanceRule
BottomNEliminatedRule
Validation Scripts
validate_generic_standings.py
validate_standings_engine.py
validate_advancement_rules.py
validate_competition_model.py
validate_stage_resolver.py
validate_competition_engine.py
validate_knockout_engine.py
validate_invitational_competition.py
Proof of Generality

The framework successfully validated a complete non-World-Cup competition:

Example Invitational Cup
    ↓
Semifinals
    ↓
Final
    ↓
Champion

This demonstrated that the project can now express competitions independently of the hardcoded World Cup 2026 simulator.

Current Limitations

Competition Framework v1 is intentionally minimal.

It does not yet support:

dynamic stage generation
automatic bracket construction
two-leg ties
best third-place qualification
draw allocation
schedule generation
home/away league seasons
full World Cup replacement
Strategic Importance

This marks the first point at which the project can credibly be described as a general football competition simulation framework, not only a World Cup simulator.

The World Cup 2026 simulator remains the production application, while the competition framework provides the reusable architectural foundation for future leagues, cups, Champions League-style formats, and custom tournaments.

Recommended Next Step

Begin a design study mapping the existing World Cup 2026 simulator onto the new competition framework.

The goal should not be immediate replacement. The goal should be to identify which parts of the current simulator can be expressed using generic competition abstractions and which missing abstractions are still required.

## Expanded Validation

Competition Framework v1 has now been validated across four distinct competition structures.

### 1. Invitational Knockout

A four-team invitational cup validated:

- knockout stages
- `Tie`
- `KnockoutEngine`
- final-stage champion inference

### 2. Domestic Cup Prototype

An eight-team domestic cup validated:

- multi-round knockout composition
- bracket construction
- quarterfinal → semifinal → final progression
- reusable `CompetitionEngine` orchestration

### 3. League Competition Prototype

An eight-team single round-robin league validated:

- `Stage(type=LEAGUE)`
- `StandingsEngine`
- standings-based champion inference
- league competition support without a league-specific engine

### 4. World Cup Group-Stage Prototype

A World Cup-style group validated:

- `Stage(type=GROUP)`
- `StandingsEngine`
- `TopNAdvanceRule`
- framework-backed representation of a World Cup group

This final validation is especially important because it reconnects the generic competition framework to the project’s flagship World Cup 2026 application.

## World Cup Framework Prototypes

Competition Framework v1 has now been validated against two World Cup-specific prototypes.

### Full Group-Stage Prototype

Study 024 validated the full 2026 World Cup group-stage structure through the generic framework.

The prototype resolved:

- 12 groups
- 72 group-stage matches
- 24 automatic top-two qualifiers

using:

- `Stage(type=GROUP)`
- `StageResolver`
- `StandingsEngine`
- `TopNAdvanceRule`

This demonstrated that the framework can express the full group-stage structure without relying on the production World Cup group-stage engine.

### Knockout Prototype

Study 025 validated a full World Cup-style knockout phase through the generic framework.

The prototype resolved:

- Round of 32
- Round of 16
- Quarterfinals
- Semifinals
- Third-place playoff
- Final

using:

- `BracketBuilder`
- `Stage(type=KNOCKOUT)`
- `Stage(type=PLAYOFF)`
- `Stage(type=FINAL)`
- `Tie`
- `MatchResult`
- `StageResolver`
- `KnockoutEngine`

This demonstrated that the framework can express the knockout half of a World Cup-style tournament without creating a World-Cup-specific knockout engine.