# Study 018 — World Cup Architecture Mapping

## Motivation

Studies 013–017 established the first version of the generic football competition framework. The framework now includes generic concepts such as `Competition`, `Stage`, `MatchResult`, `StageResolver`, `StandingsEngine`, `AdvancementRule`, and `CompetitionResult`.

The next question is whether the existing World Cup 2026 simulator can be described using this new framework.

This study does not attempt to rewrite the production World Cup simulator. Instead, it maps the current implementation onto the new competition architecture to identify what fits cleanly, what does not, and what abstractions may still be missing.

## Research Question

Can the current 2026 World Cup simulator be expressed as a composition of generic competition framework components?

## Mapping

| Existing World Cup Component | Competition Framework Concept |
|---|---|
| `simulate_tournament()` | `CompetitionEngine.resolve()` |
| `TournamentResult` | `CompetitionResult` |
| Group stage | `Stage(type=GROUP)` |
| Group standings | `StandingsEngine` |
| Group-stage qualification | `AdvancementRule` |
| Round of 32 | `Stage(type=KNOCKOUT)` |
| Round of 16 | `Stage(type=KNOCKOUT)` |
| Quarterfinals | `Stage(type=KNOCKOUT)` |
| Semifinals | `Stage(type=KNOCKOUT)` |
| Third-place playoff | `Stage(type=PLAYOFF)` |
| Final | `Stage(type=FINAL)` |
| Monte Carlo driver | Repeated `CompetitionEngine` runs plus observers |

## Clean Mappings

Several parts of the current World Cup simulator map naturally to the new framework.

The group stage maps cleanly to a standings-based stage.

Knockout rounds map naturally to staged competition phases.

`TournamentResult` is conceptually similar to `CompetitionResult`.

The observer framework from Study 012 already aligns well with generic competition results.

## Awkward Mappings

Some pieces do not yet map cleanly.

The 2026 group qualification system requires more than a simple `TopNAdvanceRule` because it includes best third-place teams.

The knockout bracket mapping is currently World Cup-specific.

The current framework does not yet have a generic `KnockoutEngine`.

The framework does not yet have a `Tie` abstraction for one-match or two-leg knockout ties.

The current framework does not yet dynamically pass qualifiers from one stage into the next.

## Missing Abstractions

This mapping suggests several future abstractions:

- `KnockoutEngine`
- `Tie`
- `Bracket`
- `BestThirdPlaceAdvanceRule`
- `StageLink` or stage transition logic
- `CompetitionSchedule`
- `DrawEngine`

## Conclusion

The existing World Cup 2026 simulator can be partially but not fully expressed using the new competition framework.

The group-stage and standings components map cleanly. The main missing pieces are knockout resolution, bracket construction, and stage-to-stage transition logic.

This confirms that the competition framework is on the right path while also identifying the next useful architectural target: a generic knockout/tie model.