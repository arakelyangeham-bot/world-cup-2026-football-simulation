COMPETITION_INGESTION_ARCHITECTURE

# Competition-Driven Ingestion Architecture

## Status

Architecture specification — Version 1

## Purpose

This document defines the canonical ingestion architecture for competition-based
football data.

The architecture supports:

- domestic leagues;
- continental club competitions;
- international tournaments;
- future competition types added to the project.

The ingestion process is organized around:

```text
Competition
    ↓
Season
    ↓
Teams
    ↓
Players
    ↓
Player Profiles
    ↓
Player-Season Statistics

The competition-season pair is the root context for all downstream football
data.

Core principle

Football data must be ingested within a defined competition and season.

A player, team, match, or statistical record should not be treated as globally
timeless when its meaning depends on the competition or season in which it was
observed.

The canonical source context is:

competition_key
+
season_start_year
+
unique_tournament_id
+
season_id

Every downstream ingestion stage must preserve enough of this context to make
the resulting data historically interpretable.

Canonical pipeline

The intended ingestion order is:

1. Register competitions
        ↓
2. Discover Sofascore seasons and season IDs
        ↓
3. Build the operational competition manifest
        ↓
4. Scrape teams
        ↓
5. Scrape players
        ↓
6. Scrape player profiles
        ↓
7. Scrape player-season statistics
        ↓
8. Audit feature coverage
        ↓
9. Build player intelligence

The order must not be reversed.

Each stage depends on identifiers or entities discovered by an earlier stage.

Layer 1 — Competition Registry
Responsibility

The Competition Registry stores stable competition identity.

Examples:

premier_league
la_liga
bundesliga
serie_a
ligue_1
champions_league
world_cup
euro

Recommended fields:

Field	Description
competition_key	Canonical internal identifier
competition_name	Human-readable display name
unique_tournament_id	Sofascore tournament identifier
competition_type	Club league, continental club, international, etc.
country_or_region	Optional competition geography
active	Whether the competition is supported

The registry should contain stable identity information only.

It should not contain:

season-specific IDs;
scrape switches;
recency weights;
feature availability;
model weights.

These belong to later layers.

Layer 2 — Season Discovery and Season Registry
Responsibility

Season discovery queries Sofascore for the seasons belonging to each registered
competition.

The discovery process resolves:

competition_key
+
season_start_year
        ↓
season_id

Recommended output fields:

Field	Description
competition_key	Canonical competition identifier
competition_name	Display name
unique_tournament_id	Sofascore tournament identifier
season_name	Sofascore season name
season_year_label	Source season label
season_start_year	Canonical numeric year
season_id	Sofascore season identifier

The existing season-discovery script already performs this role for the five
major European leagues and validates the expected competition-season grid.

The Season Registry is a discovered source registry, not an operational scrape
configuration.

Layer 3 — Competition Manifest
Responsibility

The Competition Manifest controls ingestion.

It is derived from registered and discovered competition-season records, then
enriched with operational configuration.

Recommended fields:

Field	Description
competition	Display name
competition_key	Canonical identifier
competition_type	Competition category
competition_id	Source tournament identifier
season_id	Source season identifier
season_year	Canonical season label
priority	Processing order
importance	Optional research weight
recency_weight	Time-weighting configuration
competition_importance	Competition-strength weighting
enabled	Whether the row is active
scrape_teams	Whether to ingest teams
scrape_players	Whether to ingest players
scrape_profiles	Whether to ingest profiles
scrape_stats	Whether to ingest player statistics

The Competition Manifest is an operational control table.

It should answer:

Which competition-seasons should the pipeline process, and which stages should run?

It should not be treated as the authoritative source of stable competition
identity or feature quality.

Layer 4 — Team Ingestion
Responsibility

Team ingestion establishes the valid team population for each
competition-season.

Required source inputs:

competition_id
season_id

Recommended output fields:

Field	Description
competition	Competition name
competition_type	Competition category
competition_id	Source tournament identifier
season_id	Source season identifier
season_year	Season label
team_id	Stable Sofascore team identifier
team	Team name
team_slug	Source slug

Canonical uniqueness should initially be:

competition_id
+
season_id
+
team_id

A team may appear in multiple competitions or seasons.

The ingestion process must therefore preserve competition-season membership
rather than collapsing teams into one global row.

Layer 5 — Player Ingestion
Responsibility

Player ingestion discovers players appearing in a specific competition-season.

Required source inputs:

competition_id
season_id

Recommended output fields:

Field	Description
competition	Competition name
competition_type	Competition category
competition_id	Source tournament identifier
season_id	Source season identifier
season_year	Season label
player_id	Stable Sofascore player identifier
player	Player name
player_slug	Source slug
team_id	Team associated with that player-season record
team	Team name
team_slug	Team slug

The canonical player-season-team relationship is:

competition_id
+
season_id
+
player_id
+
team_id

This relationship is the primary source for historical club membership within
the ingested competition-season.

It must not be replaced by the player's current team from the profile endpoint.

Layer 6 — Player Profile Ingestion
Responsibility

Player profile ingestion enriches stable player identity.

The profile stage occurs only after player IDs have been discovered.

Recommended profile fields include:

player_id
player
player_slug

position
positions_detailed
jersey_number

height
preferred_foot
date_of_birth

country
country_alpha2
country_alpha3

current_team_id
current_team
current_team_slug

Profile data should be interpreted carefully.

The profile endpoint supplies current descriptive information.

Therefore:

Historical team membership
    comes from the competition-season player record.

Current descriptive club
    comes from the profile endpoint.

The two should never be treated as interchangeable.

Player position information from the profile endpoint is essential for:

position normalization;
role eligibility;
expected-lineup construction;
role-specific ratings.
Layer 7 — Player-Season Statistics
Responsibility

Player-stat ingestion retrieves statistics for the exact combination:

player_id
+
competition_id
+
season_id

The output must preserve:

competition
competition_type
competition_id
season_id
season_year

player_id
player

team_id
team

before appending the returned statistical fields.

The canonical task key is:

competition_id
+
season_id
+
player_id

This enables:

resumable ingestion;
duplicate prevention;
historical player evidence;
competition-specific rating construction.
Layer 8 — Feature Coverage Manifest
Responsibility

The Feature Manifest describes data quality and availability.

It is separate from the Competition Manifest.

The Competition Manifest answers:

Should this competition-season be ingested?

The Feature Manifest answers:

Which statistical features are sufficiently available after ingestion?

Recommended fields:

competition
season_year
feature
coverage
available

Feature coverage should be calculated after player statistics are collected.

It may control later feature-engineering decisions but should not control source
discovery or entity ingestion.

Layer 9 — Player Intelligence

The ingestion outputs feed the player-intelligence pipeline:

Player Profiles
+
Player-Season Statistics
+
Feature Coverage
        ↓
Player Registry
        ↓
Attribute Scores
        ↓
Role Ratings
        ↓
Player Representation

The player registry should preserve stable player identity while retaining
competition-season evidence separately.

A single player may have evidence from:

several competitions;
several seasons;
several clubs.

These records should not be collapsed prematurely.

Historical club membership

The project already captures season-level player-team membership through the
player-ingestion output.

For a given competition-season, the relationship is:

player_id
→ team_id

This is sufficient for an initial club roster builder.

However, season-level membership may not fully resolve mid-season transfers.

For example, one player may represent two clubs in the same competition and
season.

The pipeline must therefore audit whether the source produces:

one player-team row per season

or:

multiple player-team rows for transferred players

If multiple valid team associations exist, the project will later need
date-aware membership or match-lineup evidence.

Season-level membership is acceptable for the first club infrastructure version,
provided its limitations are explicitly recorded.

Dependency rules

The following dependencies are mandatory:

Season discovery
    must occur before manifest creation.

Manifest creation
    must occur before team or player ingestion.

Player ingestion
    must occur before profile ingestion.

Player ingestion
    must occur before player-stat ingestion.

Player profiles
    must be available before role and position modeling.

Player statistics
    must be available before evidence-weighted ratings.

Team and player season membership
    must be available before club roster construction.
File responsibilities
discover_sofascore_competition_seasons.py

Responsible for:

querying Sofascore season lists;
resolving season IDs;
normalizing season start years;
validating competition-season coverage;
writing the Season Registry.
sofascore_build_competition_manifest.py

Responsible for:

combining source competition-season registries;
classifying competition types;
assigning operational weights and switches;
writing the Competition Manifest.
ingest_teams.py

Responsible for:

fetching participating teams;
preserving competition-season-team identity;
writing the team membership table.
ingest_players.py

Responsible for:

fetching players for each competition-season;
preserving the player-team relationship;
writing the player-season membership table.
ingest_player_profiles.py

Responsible for:

fetching stable and current player attributes;
supplying position information;
writing one profile per player.
ingest_player_stats.py

Responsible for:

fetching player statistics for each competition-season task;
preserving team and source context;
supporting checkpointing and resumption.
build_competition_feature_manifest.py

Responsible for:

converting measured feature coverage into availability flags;
describing downstream feature usability.
Known issues
Duplicate team extension

The current team-ingestion script extends all_teams once inside the successful
try block and again afterward.

This can duplicate successful records and may use an undefined or stale teams
variable after failure.

The post-exception extension should be removed.

Correct structure:

try:
    teams = get_teams_for_competition(row)
    all_teams.extend(teams)

except Exception as exc:
    print(...)
Manifest source inconsistency

The domestic season-discovery output and the manifest builder currently appear
to use different source filenames and column conventions.

The connection between:

sofascore_league_seasons.csv

and:

sofascore_competitions.csv

must be identified and documented.

There should be one deterministic transformation rather than manually maintained
parallel files.

Profile-stage control flag

The Competition Manifest currently includes scrape controls for teams, players,
and statistics.

A scrape_profiles field should either be added explicitly or profiles should be
documented as a global player-level stage that runs for every newly discovered
player.

Output-root consistency

All scripts should resolve file paths from the same project root convention.

No ingestion stage should accidentally write to a different nested raw
directory because OUT_DIR already points inside the data hierarchy.

Club-football extension

The club branch should consume the existing ingestion outputs in this order:

Competition-season teams
+
Competition-season players
+
Player profiles
+
Player-season statistics
        ↓
Club roster builder
        ↓
Expected starting XI
        ↓
Club Team Representation
        ↓
Club Repository
        ↓
Club expected-goals model

The first club-specific modeling divergence should occur at roster construction,
not at raw player ingestion.

Champions League support

A continental club competition should follow the same ingestion sequence:

Register Champions League
        ↓
Discover season IDs
        ↓
Enable seasons in manifest
        ↓
Ingest participating teams
        ↓
Ingest participating players
        ↓
Fetch player profiles
        ↓
Fetch competition-season statistics

A club may appear in both a domestic league and a continental competition in
the same season.

These records should remain separate evidence entries but share stable
player_id and team_id identities.

The prediction layer may later combine domestic and continental evidence using
explicit weighting rules.

Immediate implementation milestones
Milestone 1 — Fix team ingestion

Remove the duplicate all_teams.extend(teams) call and verify one unique row per:

competition_id
season_id
team_id
Milestone 2 — Audit registry-to-manifest flow

Identify exactly how the discovered domestic season registry becomes the club
input consumed by the Competition Manifest builder.

The flow must become deterministic and reproducible.

Milestone 3 — Add ingestion validation

Validate the following counts and keys:

competition-seasons
teams per competition-season
players per competition-season
profiles per unique player
statistics per player-season task
Milestone 4 — Audit transfer ambiguity

Identify player-season records associated with more than one team in the same
competition-season.

This determines whether season-level membership is sufficient for the initial
Club Roster Builder.

Milestone 5 — Build the Club Roster Builder

Only after the ingestion audit passes should the project create a club-specific
roster builder.

Acceptance criteria

The competition-driven ingestion architecture is validated when:

every active competition-season has stable source identifiers;
every active competition-season has a team population;
every player-season record preserves team identity;
every profile joins uniquely by player ID;
every statistics task preserves competition and season identity;
duplicate entity records are detected;
the registry-to-manifest transformation is reproducible;
transferred-player ambiguity is measured;
downstream club rosters can be built without using current-team profile data;
the full ingestion order is documented and repeatable.