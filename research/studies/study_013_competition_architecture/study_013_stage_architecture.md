# Stage Architecture

## Purpose

A `Stage` represents one phase of a football competition.

Examples include:

- World Cup group stage
- Champions League league phase
- Round of 16
- Quarterfinal
- Domestic league season
- Promotion playoff
- Two-leg knockout tie phase

The purpose of the `Stage` abstraction is to separate competition structure from simulation behavior.

A stage defines what kind of competition phase exists. Engines determine how that stage is simulated.

---

## Core Definition

A stage answers five questions:

1. Who participates?
2. What matches are played?
3. How are results accumulated?
4. How are teams ranked or resolved?
5. Who advances or receives placement?

---

## Proposed Stage Fields

```text
Stage
├── name
├── stage_type
├── participants
├── matches
├── advancement_rule
├── result
└── metadata

Field Definitions
name

Human-readable stage name.

Examples:

Group A
League Phase
Round of 16
Quarterfinals
Final
stage_type

The structural type of the stage.

Possible values:

group
league
knockout
two_leg_knockout
swiss
playoff
final
participants

The teams involved in the stage.

This should be a list of team names or team identifiers.

matches

The scheduled or generated matches belonging to the stage.

A stage may begin with predefined matches or rely on an engine to generate them.

advancement_rule

The rule used to determine which teams advance, qualify, are eliminated, or receive placements.

Examples:

top two advance
winner advances
top eight qualify
best third-place teams advance
aggregate winner advances
result

The completed output of the stage.

This may include:

standings table
match results
qualifiers
eliminated teams
winner
runner-up
metadata

Optional contextual information.

Examples:

group name
competition name
season
confederation constraints
home/away status
neutral-site flag
seeded status
What a Stage Should Own

A stage should own:

its identity
its participant list
its stage type
its stage-level rules
its match list or match generation context
its completed result after simulation
What a Stage Should Not Own

A stage should not:

simulate individual match scores directly
contain player ratings
contain team strength logic
know how to load repositories
write files
run Monte Carlo loops
perform post-simulation analytics

Those responsibilities belong to other layers.

Relationship to Engines

A stage is a definition.

An engine is behavior.

For example:

GroupStage
    ↓
GroupStageEngine
    ↓
StageResult

or:

LeagueStage
    ↓
StandingsEngine
    ↓
StageResult

This mirrors the broader project pattern:

Model
    ↓
Engine
    ↓
Result
Relationship to Standings

StandingsTable is not itself a stage.

It is a structure used by some stages.

For example:

group stages use standings
domestic leagues use standings
league phases use standings
knockout rounds usually do not

This distinction prevents standings logic from becoming overloaded with competition-specific behavior.

Relationship to Advancement Rules

A stage may produce rankings, winners, or standings.

An advancement rule interprets those outputs.

Examples:

StandingsTable
    ↓
TopTwoAdvanceRule
    ↓
Qualifiers
TwoLegTieResult
    ↓
AggregateWinnerRule
    ↓
Qualifier

This separation is important because different competitions may use the same stage structure but different advancement rules.

Design Principle

Stages should be composable.

A full competition should eventually be representable as:

Competition
    ├── Stage 1
    ├── Stage 2
    ├── Stage 3
    └── Stage N

The World Cup, Champions League, domestic leagues, and custom tournaments should differ by composition, not by entirely separate architecture.

Immediate Scope

For the next implementation phase, the project should support a minimal Stage data model with:

name
stage_type
participants
metadata

The first practical use case should be a standings-based stage.

This keeps the abstraction small while preparing the project for future group, league, knockout, and two-leg competition engines.