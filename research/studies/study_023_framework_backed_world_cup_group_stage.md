study_023_framework_backed_world_cup_group_stage.md

# Study 023 — Framework-backed World Cup Group Stage Prototype

## Motivation

Version 2 of the project shifts from building the Competition Framework itself to applying it to meaningful football competitions.

The first application should bridge back to the flagship project: the 2026 World Cup simulator.

The goal is not to replace the production World Cup simulator. The goal is to prove that the World Cup group stage can be expressed through the generic competition framework.

## Research Question

Can the 2026 World Cup group stage be represented using generic `Stage`, `StandingsEngine`, and `AdvancementRule` abstractions?

## Scope

This study focuses only on the group stage.

Included:

- 12 groups
- 4 teams per group
- group-stage standings
- top two qualification per group
- optional later handling of best third-place teams

Excluded:

- full Round of 32 construction
- knockout bracket mapping
- full tournament replacement
- Monte Carlo integration
- production simulator migration

## Framework Flow

```text
World Cup groups
    ↓
Stage(type=GROUP) per group
    ↓
MatchResult list per group
    ↓
StageResolver
    ↓
StandingsEngine
    ↓
StageResult
    ↓
TopNAdvanceRule
    ↓
AdvancementResult

Version 1 Plan

Start with one test group.

Then expand to all 12 groups.

Version 1 should use deterministic fake match results rather than calling the production match engine.

This keeps the study focused on framework expression, not simulation realism.

Success Criteria

Study 023 succeeds if:

A World Cup group can be represented as a generic Stage.
The group can be resolved by StageResolver.
Standings are produced by StandingsEngine.
Top two qualifiers are selected by TopNAdvanceRule.
The same pattern can scale from one group to all 12 groups.
Strategic Importance

This study reconnects the new Competition Framework to the original World Cup 2026 project.

It proves that the framework is not merely a separate toy architecture, but a potential future foundation for parts of the flagship simulator.