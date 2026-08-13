# Study 079A — Bundesliga ClubElo Integration

## Purpose

Validate complete ClubElo history coverage for all clubs contained in the Bundesliga 2024–25 production repository.

This study forms the rating-prior integration phase of the broader Bundesliga live observation study.

The production observation builder requires two independent runtime sources:

```text
ProductionClubRepository
        +
ClubEloRepository
        ↓
LiveMatchObservationBuilder

Study 078 validated the production football-intelligence repository.

Study 079A validates the ClubElo side of that boundary.

Research Question

Can every Bundesliga production club be mapped to a valid ClubElo identifier, downloaded or loaded through the existing ClubEloRepository, and resolved through the established cache contract?

Motivation

The ClubEloRepository is competition-agnostic, but ClubElo naming conventions differ from the canonical club names stored in the production repository.

Examples include:

FC Bayern München
        ↓
Bayern
Borussia Dortmund
        ↓
Dortmund
1. FC Union Berlin
        ↓
UnionBerlin

The architecture already supports these differences through a name-override mapping.

The objective was therefore not to modify the repository, but to establish a complete and validated Bundesliga alias table.

Architecture

The validated integration path is:

Bundesliga production repository
        ↓
Canonical club names
        ↓
Bundesliga ClubElo alias table
        ↓
ClubEloRepository
        ↓
Cached ClubElo histories
        ↓
Prediction-date rating resolution
Phase 1 — Cache Audit

A dedicated diagnostic was created:

audit_bundesliga_clubelo_cache.py

Its responsibilities were:

load the Bundesliga production repository;
enumerate all 18 production clubs;
apply the candidate ClubElo alias table;
derive cache filenames through the repository’s own cache_path() method;
report existing and missing histories.

Initial result:

Production clubs: 18
Existing cache files: 0
Missing cache files: 18

This established that the alias table covered all production clubs, but no Bundesliga ClubElo histories had yet been cached.

Phase 2 — History Acquisition

A dedicated acquisition script was created:

preload_bundesliga_clubelo_histories.py

The script processed every club independently and supported:

existing cache reuse;
history downloading;
repository-level normalization;
retry handling;
per-club failure isolation;
canonical resolved-name reporting.

Initial acquisition result:

Requested histories: 18
Downloaded histories: 17
Existing histories: 0
Successful histories: 17
Failed histories: 1

The only failed alias was:

Holstein Kiel
        ↓
HolsteinKiel

ClubElo returned an empty history for this identifier.

Alias Correction

The alias was corrected to:

Holstein Kiel
        ↓
Holstein

The acquisition script was rerun.

Final result:

Requested histories: 18
Downloaded histories: 1
Existing histories: 17
Successful histories: 18
Failed histories: 0
Final Bundesliga ClubElo Alias Table
1. FC Heidenheim       → Heidenheim
1. FC Union Berlin     → UnionBerlin
1. FSV Mainz 05        → Mainz
Bayer 04 Leverkusen    → Leverkusen
Borussia Dortmund      → Dortmund
Borussia M'gladbach    → Gladbach
Eintracht Frankfurt    → Frankfurt
FC Augsburg            → Augsburg
FC Bayern München      → Bayern
FC St. Pauli           → StPauli
Holstein Kiel          → Holstein
RB Leipzig             → RBLeipzig
SC Freiburg            → Freiburg
SV Werder Bremen       → Werder
TSG Hoffenheim         → Hoffenheim
VfB Stuttgart          → Stuttgart
VfL Bochum 1848        → Bochum
VfL Wolfsburg          → Wolfsburg
Final Cache Audit

The cache audit was rerun after all histories were acquired.

Result:

Production clubs: 18
Existing cache files: 18
Missing cache files: 0

All 18 mappings passed.

Cached History Coverage

The acquired histories included:

Heidenheim      926 rows
Union Berlin    2320 rows
Mainz           4424 rows
Leverkusen      5593 rows
Dortmund        6208 rows
Gladbach        5833 rows
Frankfurt       6613 rows
Augsburg        3096 rows
Bayern          5073 rows
St Pauli        4142 rows
Holstein        1904 rows
RB Leipzig      1189 rows
Freiburg        4247 rows
Werder          6866 rows
Hoffenheim      2261 rows
Stuttgart       6629 rows
Bochum          5595 rows
Wolfsburg       4354 rows

Each history was validated through the existing repository normalization logic.

This included:

required column validation;
numeric conversion;
date parsing;
interval ordering;
non-overlapping interval validation;
single resolved-club validation.
Architectural Findings

The existing ClubEloRepository required no structural changes.

It already supported:

arbitrary club histories;
disk caching;
memory caching;
retryable external acquisition through the calling script;
temporal interval resolution;
canonical ClubElo club-name validation.

The only competition-specific requirement was a Bundesliga alias table.

This is configuration data rather than architecture.

Validation Summary
Production club enumeration         PASS
Alias-table completeness            PASS
Alias-table exclusivity             PASS
Cache-path derivation               PASS
History acquisition                 PASS
Repository normalization            PASS
Canonical ClubElo resolution        PASS
Cache persistence                   PASS
18-of-18 cache coverage              PASS
Conclusion

All 18 Bundesliga production clubs now have valid and cached ClubElo histories.

The existing ClubElo repository architecture generalized without modification.

The only failed identifier, HolsteinKiel, was isolated and corrected to Holstein.

The Bundesliga runtime stack now has complete access to both required observation sources:

Production football intelligence
        +
Prediction-date ClubElo history

This completes the final evidence dependency required before constructing a Bundesliga live match observation.

Overall Result

PASS

Study 079A establishes complete Bundesliga ClubElo integration and enables the live observation construction phase.