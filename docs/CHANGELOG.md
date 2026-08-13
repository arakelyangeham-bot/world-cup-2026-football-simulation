CHANGELOG.md

Version 3 — Calibrated λ model selected.
Version 4 — Hierarchical stochastic λ sampler developed, benchmarked, and promoted to production.
Version 5 — Research initiated into low-score dependence, with the zero-score diagnostic identifying excess 0–0 draws as the dominant remaining source of scoreline error.

Study 012 — Tournament Observation Framework

• Introduced observer architecture for completed tournament simulations.
• Added reusable ObserverManager and TournamentObserver interfaces.
• Migrated aggregate tournament statistics into StatisticsObserver.
• Implemented ExtremeEventsObserver for narrative tournament analysis.
• Introduced leaderboard-based event tracking replacing single-record storage.
• Added extreme_event_leaderboards.csv output.
• Preserved complete separation between tournament simulation and tournament observation.
• Established the project's first reusable analysis layer.

## Studies 013–016 — Competition Architecture Foundation

- Defined generic competition architecture.
- Added Stage and StageResult models.
- Added generic MatchResult abstraction and adapter.
- Added StandingRow and StandingsTable.
- Implemented generic StandingsEngine.
- Added AdvancementRule architecture with TopNAdvanceRule and BottomNEliminatedRule.
- Added Competition and CompetitionResult models.
- Validated one-stage competition composition.
- Established reusable foundation for future league, group-stage, knockout, and club-football engines.

## Competition Framework v1 Established

- Added generic competition domain models.
- Added `Stage`, `StageResult`, `Competition`, and `CompetitionResult`.
- Added generic `MatchResult` abstraction and adapter.
- Added standings model and `StandingsEngine`.
- Added advancement rule architecture.
- Added `Tie` and `TieResult`.
- Added `KnockoutEngine`.
- Added `StageResolver`.
- Added minimal `CompetitionEngine`.
- Validated full non-World-Cup invitational competition.

## Competition Framework v1 Established

- Added generic competition domain models.
- Added `Competition`, `CompetitionResult`, `Stage`, `StageResult`, and `MatchResult`.
- Added `StandingRow`, `StandingsTable`, and `StandingsEngine`.
- Added `AdvancementRule`, `TopNAdvanceRule`, and `BottomNEliminatedRule`.
- Added `Tie` and `TieResult`.
- Added `KnockoutEngine`.
- Added `StageResolver`.
- Added minimal `CompetitionEngine`.
- Added `Bracket` and `BracketBuilder`.
- Validated generic standings, advancement rules, knockout stages, stage resolution, competition resolution, bracket construction, invitational cup, and domestic cup prototype.
- Established the project’s first reusable football competition framework independent of the World Cup 2026 simulator.

## Competition Framework v1 + World Cup Group Prototype

- Established reusable football competition architecture.
- Added generic competition models, stage models, match results, standings, ties, brackets, and advancement rules.
- Added `StandingsEngine`, `KnockoutEngine`, `StageResolver`, and `CompetitionEngine`.
- Validated invitational knockout competition.
- Validated domestic cup prototype.
- Validated 8-team league prototype.
- Validated framework-backed World Cup group-stage prototype.
- Confirmed that the framework can express cups, leagues, and World Cup-style groups without competition-specific engines.

## Competition Framework v1 and World Cup Prototypes

- Established reusable football competition framework.
- Added generic models for competitions, stages, match results, standings, ties, brackets, and results.
- Added `StandingsEngine`, `KnockoutEngine`, `StageResolver`, and `CompetitionEngine`.
- Added advancement rule architecture.
- Added bracket construction.
- Validated invitational, domestic cup, and league competition prototypes.
- Validated full framework-backed World Cup group-stage prototype.
- Validated framework-backed World Cup knockout prototype.
- Established foundation for a complete framework-backed World Cup prototype.

### Study 077 — Bundesliga Football Intelligence Integration

Validated the complete football intelligence pipeline for Bundesliga 2024–25. Expanded the player-evidence ingestion framework, regenerated the processed player-intelligence artifacts, and successfully constructed a runtime `TeamRepresentation` for FC Bayern München without introducing Bundesliga-specific architecture. This study demonstrates that the Version 1 football intelligence architecture generalizes to additional domestic competitions through evidence expansion rather than architectural redesign.

### Study 078 — Bundesliga Production Repository

Introduced a reusable production club-repository framework that serializes competition-aware `TeamRepresentation` objects into deterministic runtime CSV artifacts. Built and validated `bundesliga_club_repository_v1.csv` for all 18 Bundesliga 2024–25 clubs and confirmed successful reload through the existing `ProductionClubRepository` without runtime architecture changes.

### Study 079A — Bundesliga ClubElo Integration

Created a Bundesliga ClubElo cache audit and history-acquisition workflow. Defined and validated aliases for all 18 Bundesliga 2024–25 clubs, acquired complete ClubElo histories, corrected the Holstein Kiel alias from `HolsteinKiel` to `Holstein`, and confirmed 18-of-18 cache coverage through the existing competition-agnostic `ClubEloRepository`.

## Version 2A Architecture Review and Production Freeze

- Completed a full repository and dependency review after Study 084.
- Confirmed the authoritative club prediction runtime is `ProductionPredictionPipeline`.
- Confirmed both active club and national-team simulators remain scoreline-first.
- Verified that club simulation already integrates the production goal model through the football-model adapter and existing scoreline sampler.
- Classified active, authoritative, compatibility, research and legacy runtime paths.
- Documented the Version 2A architecture, runtime lifecycle and retrospective.
- Approved the Version 2A production freeze.
- Established Version 2B as the scientific model-improvement phase.



Milestone

Modular Representation Architecture Complete

Summary

- Feature transformations are now interchangeable.
- Player attribute generation is configurable.
- Player rating generation is configurable.
- Repository generation supports multiple representations.
- Canonical production behavior is preserved.
- Three matched Bundesliga repository branches generated.
- Architecture now supports controlled representation experiments.