study_019_bracket_architecture.md

# Study 019 — Bracket Architecture

## Motivation

Competition Framework v1 introduced generic stages, standings, advancement rules, ties, knockout resolution, and competition composition. However, knockout stages currently require ties to be manually defined.

To support reusable cups, playoffs, Champions League-style knockouts, and future World Cup framework prototypes, the project needs a generic way to turn qualified teams into knockout ties.

Study 019 introduces bracket architecture.

## Research Question

How should the framework construct knockout ties from a list of qualified teams while remaining independent of any specific competition format?

## Core Concept

A bracket is an ordered structure that pairs teams into knockout ties.

The bracket should not simulate matches.

The bracket should only answer:

```text
Which teams play each other?

Simulation is still handled by match engines or by completed MatchResult objects.

Resolution is still handled by KnockoutEngine.

Proposed Abstractions
Bracket

A collection of ties belonging to a knockout stage.

BracketBuilder

A component that turns teams into ties.

Seeding Strategy

A rule that determines how teams are ordered or paired.

Examples:

fixed order
seed high vs low
random draw
group winner vs runner-up constraints
country/confederation protection
Version 1 Scope

Study 019 v1 should be intentionally small.

It should support:

fixed-order brackets
high-seed vs low-seed pairing
even number of teams
single-match ties only

It should not yet support:

random draws
two-leg ties
byes
reseeding
country protection
host constraints
group rematch avoidance
World Cup-specific bracket mapping
Example

Input:

1. Argentina
2. Brazil
3. Morocco
4. Japan

Output:

Argentina vs Japan
Brazil vs Morocco

This corresponds to:

1 vs 4
2 vs 3
Relationship to Existing Framework

The bracket builder produces Tie objects.

Those ties can be assigned to a Stage(type=KNOCKOUT).

The stage can then be resolved by StageResolver, which dispatches to KnockoutEngine.

Qualified teams
    ↓
BracketBuilder
    ↓
Tie list
    ↓
Stage(type=KNOCKOUT)
    ↓
StageResolver
    ↓
KnockoutEngine
    ↓
StageResult
Design Principle

Bracket construction should be separate from knockout resolution.

A bracket determines matchups.

A knockout engine determines winners from completed ties.

This prevents bracket logic, match simulation, and advancement logic from becoming entangled.

Initial Success Criteria

Study 019 v1 succeeds if the project can:

Accept an ordered list of teams.
Generate valid knockout ties.
Attach those ties to a knockout stage.
Resolve the stage using the existing KnockoutEngine.
Validate the result through a small script.