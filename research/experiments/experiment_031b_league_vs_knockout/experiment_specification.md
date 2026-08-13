# Experiment 031B — League vs Knockout

## Research Question

Does a league competition identify the strongest team more reliably than a knockout competition?

## Motivation

A league gives teams repeated matches and should reduce the influence of single-match randomness. A knockout tournament is more volatile because one poor match can eliminate even the strongest team.

Experiment 031B is the first true Version 3 football research experiment. It compares two competition formats while holding team strengths, metrics, and synthetic match-generation logic fixed.

## Fixed Variables

- Synthetic 8-team strength ladder
- Team strengths
- Match-result generation method
- Simulation count
- Random seed policy
- Metric library
- Experiment runner

## Independent Variable

Competition format:

1. Single round-robin league
2. Single-elimination knockout tournament

## Dependent Variables

- average champion strength
- strongest-team championship rate
- champion variance
- upset rate

## Hypothesis

League competitions will identify the strongest team more reliably than knockout competitions because repeated matches reduce the influence of single-match variance.

## Experimental Design

Run both formats repeatedly using the same synthetic team pool.

### Team Strengths

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

### League Format

- 8 teams
- single round-robin
- 28 matches
- champion determined by final standings

### Knockout Format

- 8 teams
- quarterfinals
- semifinals
- final
- high-vs-low initial seeding
- champion determined by final winner

## Version 1 Scope

This experiment uses synthetic teams and synthetic match generation.

Player Intelligence repositories are intentionally excluded from v1 so competition-format effects are not mixed with team-representation effects.

## Success Criteria

Experiment 031B succeeds if it produces comparable metric reports for league and knockout formats and supports a football interpretation of the differences.