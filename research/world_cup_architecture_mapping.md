world_cup_architecture_mapping.md

# World Cup 2026 Architecture Mapping

## Purpose

This document maps the existing World Cup 2026 simulator onto the new Competition Framework v1.

The goal is not to replace the production World Cup simulator yet. The goal is to understand which parts of the current simulator already correspond to generic competition abstractions and which parts still require future framework development.

## Current Production Flow

The existing World Cup simulator follows this structure:

```text
simulate_tournament()
    ↓
simulate_group_stage_with_matches()
    ↓
extract_qualifiers_from_standings()
    ↓
build_round_of_32()
    ↓
simulate knockout rounds
    ↓
TournamentResult

Framework Mapping
Current World Cup Component	Competition Framework v1 Concept	Mapping Status
simulate_tournament()	CompetitionEngine.resolve()	Partial
TournamentResult	CompetitionResult	Partial
Group stage	Stage(type=GROUP)	Clean
Group-stage standings	StandingsEngine / StandingsTable	Clean
Group-stage qualifiers	AdvancementRule	Partial
Round of 32	Stage(type=KNOCKOUT)	Conceptual
Round of 16	Stage(type=KNOCKOUT)	Conceptual
Quarterfinals	Stage(type=KNOCKOUT)	Conceptual
Semifinals	Stage(type=KNOCKOUT)	Conceptual
Third-place playoff	Stage(type=PLAYOFF)	Clean
Final	Stage(type=FINAL)	Clean
Knockout matches	Tie / KnockoutEngine	Partial
Monte Carlo driver	repeated CompetitionEngine runs + observers	Conceptual
Clean Mappings
Group Stage

The World Cup group stage maps naturally to:

Stage(type=GROUP)
    ↓
StandingsEngine
    ↓
StageResult

The generic standings model already supports wins, draws, losses, goals for, goals against, goal difference, and points.

Knockout Rounds

Each knockout round maps naturally to:

Stage(type=KNOCKOUT)
    ↓
KnockoutEngine
    ↓
StageResult

The current framework already supports single-match knockout ties through Tie, TieResult, and KnockoutEngine.

Final

The World Cup final maps naturally to:

Stage(type=FINAL)
    ↓
KnockoutEngine
    ↓
CompetitionResult.champion

This was validated through the invitational competition prototype.

Partial Mappings
Group Qualification

The current World Cup uses:

top two teams from each group
+
best third-place teams

Competition Framework v1 supports TopNAdvanceRule, but not yet best third-place selection across groups.

Missing abstraction:

BestThirdPlaceAdvanceRule

or a more general cross-stage / cross-table advancement rule.

Knockout Bracket Construction

The current simulator has World Cup-specific bracket construction logic.

Competition Framework v1 has Tie and KnockoutEngine, but does not yet have a generic Bracket or BracketBuilder.

Missing abstractions:

Bracket
BracketBuilder
StageLink
Stage-to-Stage Transitions

The current framework resolves stages in order, but it does not yet dynamically pass qualifiers from one stage into the next.

Missing abstraction:

StageTransition

or:

StageLink
What Should Not Be Rewritten Yet

The production World Cup simulator should remain unchanged for now.

It is already validated, integrated with Monte Carlo simulations, and connected to existing output/reporting workflows.

The new framework should mature alongside it before replacing any production logic.

Recommended Next Framework Targets
BestThirdPlaceAdvanceRule
Bracket / BracketBuilder
StageTransition
Framework-backed mini World Cup prototype
Optional future migration of World Cup group-stage logic
Conclusion

The existing World Cup 2026 simulator maps well to the new Competition Framework at the conceptual level.

The group stage, standings, knockout rounds, playoff, and final all have clear framework equivalents.

The main missing pieces are cross-group qualification, bracket construction, and dynamic stage transitions.

This confirms that Competition Framework v1 is directionally correct while also identifying the next set of abstractions needed before any production World Cup migration should be attempted.