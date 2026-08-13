# Competition Engine Generalization Plan

## Thesis

The next major phase of the World Cup 2026 Data Science Project should generalize the simulator from a World Cup-specific tournament system into a reusable football competition simulation framework.

The 2026 World Cup simulator should remain the flagship production application, but the surrounding architecture should become capable of supporting other competition formats such as leagues, group stages, two-leg knockout ties, custom tournaments, and future club competitions.

## Strategic Goal

Move from:

“World Cup 2026 simulator”

to:

“Football competition simulation framework, with World Cup 2026 as the flagship case study.”

## Guiding Principles

1. Preserve the existing World Cup 2026 simulator.
2. Do not rewrite working tournament code prematurely.
3. Add generic competition engines beside the existing production engine.
4. Keep match simulation separate from competition structure.
5. Build reusable abstractions only when they answer real football questions.
6. Avoid format sprawl.
7. Prioritize small, testable modules.

## Phase 1: Monte Carlo Event Tracker

This should be the first module.

Purpose:

Capture extreme and interesting events across tournament simulations.

Examples:

* Biggest blowout
* Biggest underdog win
* Highest-scoring match
* Highest-scoring draw
* Most goals scored by a champion
* Fewest goals conceded by a champion
* Strongest team eliminated in the group stage
* Weakest team reaching a semifinal
* Most chaotic group
* Lowest-points qualifier
* Most dominant tournament run

This should behave as an observer layer, not as part of match or tournament logic.

Strategic value:

It makes the simulator tell stories, not only probabilities.

## Phase 2: Generic Standings Engine

Purpose:

Create reusable league/group-table logic.

Should support:

* Matches played
* Wins
* Draws
* Losses
* Goals for
* Goals against
* Goal difference
* Points
* Ranking rules
* Tiebreakers

This becomes the foundation for:

* World Cup groups
* domestic leagues
* Champions League league phase
* custom group stages

Important boundary:

Do not build separate table logic for every competition.

## Phase 3: Generic Knockout Engine

Purpose:

Separate the idea of a match from the idea of a tie.

A knockout tie may consist of:

* one match
* two legs
* aggregate score
* extra time
* penalties
* optional away-goals logic if modeling historical formats

This unlocks:

* World Cup knockouts
* domestic cups
* Champions League knockouts
* Europa League-style formats
* qualification playoffs

## Phase 4: Draw and Allocation Engine

Purpose:

Support tournament construction rather than only tournament simulation.

Should eventually support:

* pots
* groups
* seeded draws
* host constraints
* confederation constraints
* repeated draw attempts
* custom tournament sizes

Boundary:

This should not replace the hardcoded 2026 World Cup bracket. The official 2026 simulator remains the production implementation. The generic draw engine is reusable infrastructure.

## Postponed Ideas

These are interesting but should wait:

* LLM player bios
* player aging
* tactical chemistry
* injuries
* dynamic form
* full club football expansion
* broader league scraping

These should return later once competition architecture is stronger.

## Recommended Immediate Next Step

Begin with the Monte Carlo Event Tracker.

Reason:

It is low-risk, fun, modular, and directly improves the current simulator without requiring a major architectural rewrite.

The first implementation should answer:

“What were the most extreme things that happened across 10,000 simulated World Cups?”

## Success Criteria

This phase is successful if the project can:

1. Run the existing World Cup simulator unchanged.
2. Track extreme tournament events across Monte Carlo runs.
3. Produce readable summary outputs.
4. Add narrative insight without modifying match simulation logic.
5. Establish the first reusable observer-style module for future competition engines.

## Long-Term Vision

Eventually, the project should support:

* World Cup 2026
* custom international tournaments
* domestic leagues
* Champions League-style formats
* two-leg knockout competitions
* simulation research studies across formats

The long-term identity should be:

A modular football competition simulation framework with player-informed team representations, probabilistic match generation, and reusable competition engines.