pre_match_observatory_architecture.md

# Pre-Match Observatory Architecture

## Purpose

The Pre-Match Observatory is the first version of the Football Observatory.

It studies what can be learned from:

```text
pre-match team information
  ↓
final match outcome
```

using the historical training dataset currently available in the project.

## Scope

Version 1 focuses only on data that is already available:

```text
team-strength features
Poisson attack / defense features
FIFA points
final scoreline
match result
```

It does not require:

```text
event data
shots
xG events
possession
cards
substitutions
goal timings
formations
lineups
injuries
```

Those may be added later.

## Core Architecture

```text
Historical Training Dataset
  ↓
PreMatchObservation
  ↓
ObservedMatchOutcome
  ↓
Football Observable Metrics
  ↓
Football Gap Report
```

## Match Observation Contract

A match observation should eventually support both pre-match and in-match data.

Conceptually:

```python
MatchObservation(
    prematch=PreMatchObservation(...),
    outcome=ObservedMatchOutcome(...),
    events=None,
)
```

Later, when event-level data becomes available:

```python
MatchObservation(
    prematch=PreMatchObservation(...),
    outcome=ObservedMatchOutcome(...),
    events=InMatchEventLog(...),
)
```

The important rule is:

```text
events must be optional
```

The Observatory should work with pre-match data alone.

## PreMatchObservation

Fields should include the current historical training dataset features:

```text
home_attack
home_midfield
home_defense
home_gk
away_attack
away_midfield
away_defense
away_gk

attack_diff
midfield_diff
defense_diff
gk_diff

home_poisson_attack
home_poisson_defense
away_poisson_attack
away_poisson_defense

poisson_attack_diff
poisson_defense_diff

home_fifa_points
away_fifa_points
fifa_points_diff
```

## ObservedMatchOutcome

Fields should include:

```text
home_score
away_score
result
total_goals
goal_difference
is_draw
is_home_win
is_away_win
is_one_goal_match
is_clean_sheet
both_teams_scored
is_high_scoring
is_blowout
scoreline
```

These can be derived from the final scoreline.

## Version 1 Research Questions

The Pre-Match Observatory should answer:

```text
How do final scorelines vary with pre-match team strength?
How does draw probability vary with FIFA points difference?
How do one-goal matches vary with team-strength gap?
How does total-goal frequency vary with attack/defense differences?
How often do both teams score under different matchup profiles?
How often do clean sheets occur under different mismatch sizes?
```

## Conditional Football

The main scientific focus of Version 1 should be conditional football.

Not just:

```text
What is the draw rate?
```

But:

```text
What is the draw rate when the teams are evenly matched?
What is the draw rate when one team is much stronger?
```

Not just:

```text
How often do 2-1 matches occur?
```

But:

```text
How often do 2-1 matches occur in balanced matches?
How often do 2-1 matches occur when the home/team1 side is stronger?
```

## Suggested Binning Dimensions

The first version should support bins such as:

```text
fifa_points_diff_bin
attack_diff_bin
poisson_attack_diff_bin
poisson_lambda_diff_bin
team_strength_gap_bin
```

Version 1 should start with FIFA points difference and Poisson lambda difference because they are interpretable and already production-relevant.

## Future In-Match Extension

When event-level data becomes available, the Observatory can add:

```text
goal timing
shot timing
xG timing
cards
substitutions
possession phases
game state transitions
```

This would enable process-level questions such as:

```text
Do leading teams reduce tempo?
Do trailing teams increase shot volume?
Are late goals structurally different from early goals?
Do substitutions alter scoring rates?
```

## Design Rule

Do not block current research on future data.

The Observatory should be useful now with pre-match observations and final outcomes.

Future event data should extend the system, not force a redesign.

## Phase Definition

```text
Phase 5A: Pre-Match Football Observatory
Phase 5B: In-Match Football Observatory
```

## Guiding Principle

Work with the data we have now, while designing the architecture so that richer football data can be plugged in later.
