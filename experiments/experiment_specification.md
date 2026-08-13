experiment_specification.md

# Experiment 031 — League vs Knockout

## Research Question

Does a league competition identify the strongest team more reliably than a knockout competition?

## Motivation

Football competitions reward different qualities.

A league gives teams repeated opportunities and should reduce the effect of single-match randomness. A knockout tournament is more volatile because one bad match can eliminate even the strongest team.

This experiment compares the two formats under controlled conditions.

## Fixed Variables

- Team set
- Team strengths
- Match engine
- Goal model
- Simulation count
- Random seed policy
- Metrics
- Observer logic

## Independent Variable

Competition format.

Two formats are compared:

1. Single round-robin league
2. Single-elimination knockout tournament

## Synthetic Team Set

| Team | Strength |
|---|---:|
| Team A | 100 |
| Team B | 95 |
| Team C | 90 |
| Team D | 85 |
| Team E | 80 |
| Team F | 75 |
| Team G | 70 |
| Team H | 65 |

## Dependent Variables

- Strongest-team championship rate
- Average champion strength
- Champion variance
- Upset rate

## Hypothesis

League competitions will identify the strongest team more reliably than knockout competitions because repeated matches reduce the influence of single-match variance.

## Experimental Design

Run the same eight-team field through two competition formats.

### League Format

- 8 teams
- single round-robin
- 28 matches
- champion determined by standings

### Knockout Format

- 8 teams
- quarterfinals
- semifinals
- final
- champion determined by final winner

## Version 1 Scope

This experiment uses synthetic team strengths only.

Player Intelligence repositories are intentionally excluded from v1 to avoid mixing competition-format effects with team-representation effects.

## Future Extensions

- v2: repeat using `dimension-specific` Player Intelligence repository.
- v3: compare results across repository sources.
- v4: test seeded vs random knockout brackets.

## Success Criteria

Experiment 031 v1 succeeds if it produces a clear comparison between league and knockout formats using the same teams, same strengths, and same metric definitions.