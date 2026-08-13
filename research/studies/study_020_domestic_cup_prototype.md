study_020_domestic_cup_prototype.md

# Study 020 — Domestic Cup Prototype

## Motivation

Competition Framework v1 established reusable competition abstractions including stages, standings, advancement rules, ties, knockout engines, and bracket construction.

The next step is to prove that these abstractions can express a recognizable competition format beyond both the World Cup 2026 simulator and toy validation scripts.

Study 020 uses a simple domestic cup structure as the first real competition prototype.

## Research Question

Can the generic competition framework express an 8-team knockout cup without creating competition-specific engine logic?

## Format

The prototype cup uses:

- 8 teams
- Quarterfinals
- Semifinals
- Final
- Single-match knockout ties
- Fixed high-vs-low bracket seeding

## Scope

This study does not model:

- real cup schedules
- home/away advantage
- replays
- two-leg ties
- random draws
- seeded draw constraints
- lower-division teams
- player rotation
- squad fatigue

The purpose is architectural validation, not realism.

## Expected Framework Flow

```text
Ordered team list
    ↓
BracketBuilder
    ↓
Quarterfinal ties
    ↓
Stage(type=KNOCKOUT)
    ↓
KnockoutEngine
    ↓
Semifinalists
    ↓
BracketBuilder
    ↓
Semifinal ties
    ↓
KnockoutEngine
    ↓
Finalists
    ↓
Final stage
    ↓
Champion

Success Criteria

Study 020 succeeds if:

An 8-team cup can be represented as a Competition.
Each round can be represented as a Stage.
Each knockout round can be resolved by KnockoutEngine.
The final champion is stored in CompetitionResult.
No domestic-cup-specific engine is created.
Strategic Importance

This study demonstrates that the project is no longer limited to World Cup-style tournament logic.

A domestic cup prototype proves that the Competition Framework can express a second recognizable competition family using the same reusable abstractions.