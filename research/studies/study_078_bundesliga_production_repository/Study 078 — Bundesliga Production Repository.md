# Study 078 — Bundesliga Production Repository

## Purpose

Validate that the competition-aware football intelligence layer can be promoted into a stable runtime production artifact for Bundesliga 2024–25 without modifying the existing runtime repository architecture.

This study follows:

- Study 076 — Big Five Player-Evidence Pipeline Generalization
- Study 077 — Bundesliga Football Intelligence Integration

The previous study proved that the existing football-intelligence architecture could construct valid Bundesliga `TeamRepresentation` objects.

The next architectural question was:

> Can those representations be serialized into a deterministic production repository that is accepted by the existing runtime `ProductionClubRepository`?

---

# Research Question

Can the existing Bundesliga football intelligence be transformed into a reusable production repository containing all 18 Bundesliga clubs, while preserving the runtime repository contract already used for Premier League prediction?

---

# Motivation

The production observation pipeline does not consume `CompetitionTeamRepository` directly.

Instead, it consumes a stable CSV artifact through:

```text
ProductionClubRepository

The existing runtime artifact had been produced specifically for the Premier League during Study 071A.

Therefore, the missing architectural layer was:

CompetitionTeamRepository
        ↓
Production repository builder
        ↓
Competition-specific production CSV
        ↓
ProductionClubRepository

The objective of this study was to generalize this transformation without:

modifying Study 071A;
duplicating the logic in a Bundesliga-specific implementation;
introducing prediction or ClubElo responsibilities into the builder;
changing the existing runtime repository loader.
Architecture

The implemented production pipeline is:

CompetitionPlayerRepository
        ↓
CompetitionRosterBuilder
        ↓
CompetitionTeamRepository
        ↓
TeamRepresentation
        ↓
ProductionClubRepositoryBuilder
        ↓
bundesliga_club_repository_v1.csv
        ↓
ProductionClubRepository
Implementation

A new reusable production package was introduced.

research/production/
    __init__.py
    production_repository_schema.py
    production_repository_config.py
    production_club_repository_builder.py

A dedicated Study 078 orchestration script was also created:

research/studies/
    study_078_bundesliga_production_repository/
        build_bundesliga_production_repository.py
Production Repository Contract

Each club record contains the full football-intelligence representation:

club

attack
midfield
defense
goalkeeper

attack_depth
midfield_depth
defense_depth

squad_quality
evidence_score

representation_type
aggregation_profile

player_count
available_player_count

repository_version
repository_scope
representation_season_id

The artifact intentionally contains more information than the current prediction model requires.

This preserves the complete football-intelligence snapshot and allows future models to use additional representation dimensions without changing the production repository schema.

Builder Responsibilities

The generic ProductionClubRepositoryBuilder is responsible for:

consuming club names;
receiving a representation-provider callable;
serializing TeamRepresentation objects;
validating record integrity;
validating persistence integrity;
writing a deterministic CSV artifact.

It is explicitly not responsible for:

computing player intelligence;
constructing squads;
resolving ClubElo;
building observations;
fitting models;
generating predictions;
simulating matches.

This preserves separation between domain intelligence and production persistence.

Bundesliga Configuration

Target competition:

Bundesliga

Target season:

2024–25

Resolved identifiers:

Competition ID: 35
Season ID: 63516

Expected club count:

18

Representation type:

full_squad

Repository version:

v1

Repository scope:

bundesliga_2024_25

Output artifact:

outputs/
    study_078_bundesliga_production_repository/
        bundesliga_club_repository_v1.csv
Validation Results

The study successfully resolved all 18 Bundesliga clubs.

1. FC Heidenheim
1. FC Union Berlin
1. FSV Mainz 05
Bayer 04 Leverkusen
Borussia Dortmund
Borussia M'gladbach
Eintracht Frankfurt
FC Augsburg
FC Bayern München
FC St. Pauli
Holstein Kiel
RB Leipzig
SC Freiburg
SV Werder Bremen
TSG Hoffenheim
VfB Stuttgart
VfL Bochum 1848
VfL Wolfsburg

Repository validation:

Repository rows: 18
Unique clubs: 18
Runtime clubs: 18
Missing required values: 0
Representation type: full_squad

The generated artifact reloaded successfully through the existing:

ProductionClubRepository

No modifications were required to the runtime loader.

Diagnostic Observation

The following serialized fields were zero for every club:

squad_quality
evidence_score

This does not represent a Study 078 failure.

The builder correctly serialized the values returned by the underlying TeamRepresentation objects.

The current production model does not use these fields.

The active model feature contract uses:

home_attack
away_attack
home_defense
away_defense
attack_depth_diff
rating_prior_diff

Therefore, the zero values do not block production observation or prediction integration.

This should be retained as a non-blocking football-intelligence diagnostic for future investigation.

Architectural Findings

The production repository layer successfully generalized beyond the Premier League.

No Bundesliga-specific logic was introduced into:

ProductionClubRepositoryBuilder
ProductionClubRecord
ProductionRepositoryConfig
ProductionClubRepository

The only Bundesliga-specific component is the study-level configuration that identifies:

competition;
season;
expected club count;
output scope.

This confirms that the production artifact layer is competition-agnostic.

Validation Summary
Competition-season resolution       PASS
Club enumeration                    PASS
Expected club count                 PASS
TeamRepresentation construction     PASS
Serialization                       PASS
Record validation                   PASS
DataFrame validation                PASS
Deterministic CSV generation        PASS
Runtime repository reload           PASS
Runtime club resolution             PASS
Conclusion

The Bundesliga football-intelligence layer was successfully promoted into a stable runtime production artifact.

The resulting repository contains all 18 Bundesliga clubs and is fully compatible with the existing ProductionClubRepository.

No changes were required to the runtime architecture.

The study demonstrates that the production repository pattern introduced during Study 071A can be generalized across competitions through configuration and dependency injection rather than duplication or redesign.

Overall Result

PASS

Study 078 establishes the production artifact boundary required for Bundesliga live observation and prediction integration.