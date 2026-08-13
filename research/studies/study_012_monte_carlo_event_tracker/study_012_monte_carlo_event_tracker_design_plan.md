# Study 012: Monte Carlo Event Tracker

## Core Question

What are the most extreme, surprising, and narratively meaningful events that occur across repeated World Cup simulations?

## Thesis

The Monte Carlo simulator should not only estimate probabilities. It should also surface the stories hidden inside the simulation distribution.

Instead of only asking:

“Who wins most often?”

Study 012 asks:

“What kinds of strange, dramatic, dominant, or chaotic tournaments can this model produce?”

## Strategic Purpose

This study adds a narrative layer to the simulator without changing match simulation, team strength, player intelligence, or tournament logic.

It should behave as an observer system.

The simulator generates tournaments.

The event tracker watches them.

## First Event Categories

### Match-level extremes

* Biggest blowout
* Highest-scoring match
* Highest-scoring draw
* Biggest underdog win
* Biggest favorite collapse
* Most surprising knockout upset

### Team tournament extremes

* Most goals scored by a champion
* Fewest goals conceded by a champion
* Most dominant champion run
* Weakest champion by pre-tournament strength
* Strongest team eliminated in group stage
* Weakest team reaching quarterfinals/semifinals/final

### Group-stage extremes

* Most chaotic group
* Lowest-points qualifier
* Highest-points eliminated team
* Largest goal-difference swing
* Group with most total goals
* Group with most draws

### Knockout-stage extremes

* Highest-scoring knockout match
* Biggest knockout blowout
* Most penalty shootouts in one tournament
* Champion with most extra-time/penalty wins
* Deepest underdog run

## Important Design Principle

Do not track everything immediately.

Start with a small, high-value set:

1. Biggest blowout
2. Highest-scoring match
3. Highest-scoring draw
4. Biggest underdog win
5. Strongest group-stage elimination
6. Weakest semifinalist
7. Most goals by a champion
8. Fewest goals conceded by a champion
9. Most dominant champion
10. Most chaotic group

That is enough for Study 012 v1.

## Required Data Per Simulated Tournament

The event tracker probably needs access to:

* tournament ID / simulation ID
* match records
* teams
* goals for and against
* stage
* group
* winner
* loser
* draw status
* knockout progression
* champion
* final placements
* pre-tournament team strength
* possibly team repository values

## Architectural Boundary

The event tracker should not decide match outcomes.

It should not alter standings.

It should not influence advancement.

It should only observe completed simulations.

Ideal structure:

`Tournament simulation -> Tournament result object -> Event tracker -> Event summary`

## Output Ideas

Study 012 should produce:

* `monte_carlo_event_summary.csv`
* `monte_carlo_extreme_matches.csv`
* `monte_carlo_extreme_teams.csv`
* `monte_carlo_extreme_groups.csv`
* `study_012_monte_carlo_event_tracker.md`

## Scientific Value

This study helps evaluate whether the simulator produces plausible football worlds.

For example:

* Are the biggest blowouts believable?
* Are the weakest semifinalists plausible?
* Are group-stage eliminations realistic?
* Does the model generate enough chaos?
* Does the champion profile look too dominant, too random, or appropriately varied?

This becomes another validation lens beyond scoreline TVD and champion probabilities.

## Portfolio Value

This is highly presentable.

Probability tables are useful, but event stories are memorable.

Examples:

* “In 10,000 simulations, the biggest upset was New Zealand beating France 3–1.”
* “The most dominant champion scored 24 goals and conceded 2.”
* “The strongest team eliminated in the group stage was Brazil.”
* “The most chaotic group had all four teams finish on 4 points.”

These are the kinds of outputs people actually enjoy reading.

## Success Criteria

Study 012 succeeds if it:

1. Runs alongside the current simulator.
2. Requires minimal changes to existing code.
3. Produces interpretable extreme-event outputs.
4. Adds narrative value.
5. Helps diagnose whether the simulator’s football universe feels plausible.
6. Creates a foundation for future dashboards and reports.

## Recommended v1 Scope

The first version should not try to be exhaustive.

Build the observer around three categories:

1. Match extremes
2. Team tournament extremes
3. Group-stage extremes

Postpone advanced knockout narratives until v2.

## Study 012 v1 Event List

### Match Extremes

* Biggest blowout
* Highest-scoring match
* Highest-scoring draw
* Biggest underdog win

### Team Extremes

* Most goals by champion
* Fewest goals conceded by champion
* Weakest semifinalist
* Strongest group-stage elimination

### Group Extremes

* Most chaotic group
* Lowest-points qualifier

That is the right starting set.
