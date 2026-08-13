STUDY_046_CLUB_MEMBERSHIP_AUDIT

# Study 046 — Competition-Season Club Membership Audit

## Status

Research design specification

## Research question

Does the existing competition-season ingestion pipeline contain sufficiently
complete and internally consistent player-to-club membership information to
construct historically valid club squads?

---

# Motivation

The club-football branch requires a reliable way to answer:

> Which players belonged to a particular club in a particular competition and
> season?

The existing player-ingestion output already records:

```text
competition
competition_type
competition_id
season_id
season_year

team_id
team

player_id
player

This suggests that historical club membership may already be recoverable from
the ingestion pipeline without relying on the current_team field from player
profiles.

Before implementing a Competition Player Repository or Club Roster Builder, the
membership data must be audited.

Scope

Study 046 evaluates the membership information contained in:

data/raw/sofascore/sofascore_players.csv

The study does not yet:

build club rosters;
select expected starting lineups;
calculate club team representations;
train a club expected-goals model;
resolve transfers using match-level data;
build a league or Champions League simulator.

Its purpose is only to determine whether the current ingestion output is a
sufficient foundation for those later steps.

Unit of analysis

The fundamental membership record is:

competition_id
+
season_id
+
team_id
+
player_id

The primary context is:

competition
+
season

Within that context, each player should normally be associated with one team.

Multiple valid team memberships may occur because of mid-season transfers.

Hypotheses
H1 — Competition-season completeness

Every enabled competition-season contains:

at least one team;
at least one player;
non-null competition and season identifiers.

Expected outcome:

No empty competition-seasons.
H2 — Team membership completeness

Every player membership record contains:

team_id
team
player_id
player

Expected outcome:

Missing identifier and name rates are negligible or zero.
H3 — Stable team identity

Within one competition-season, each team_id maps to exactly one normalized
team name.

Expected outcome:

team_id → one team name

Name changes caused only by harmless punctuation, accents, or source formatting
should be documented separately from genuine identity conflicts.

H4 — Stable player identity

Within one competition-season, each player_id maps to exactly one player name.

Expected outcome:

player_id → one player name

Minor source-name variations should be distinguished from genuine player-ID
conflicts.

H5 — Plausible club squad sizes

The number of unique players associated with each club should resemble a
plausible season player population.

The study will not impose a rigid football-law squad limit because Sofascore
season statistics may include:

youth players;
short-term registrations;
transferred players;
players with very limited minutes;
reserve players who appeared in the competition.

Initial diagnostic thresholds:

suspiciously small: fewer than 15 unique players
suspiciously large: more than 60 unique players

These thresholds are audit flags, not automatic failures.

H6 — Duplicate row control

Exact duplicate membership rows should be absent after applying the canonical
membership key:

competition_id
season_id
team_id
player_id

Expected outcome:

Exact duplicate count = 0

If duplicates exist, they should be removable without losing information.

H7 — Transfer ambiguity is measurable

Players associated with multiple clubs in the same competition-season should be
identifiable.

These records are not automatically errors.

They may represent:

mid-season transfers;
loans;
source-data duplication;
team-ID changes;
erroneous joins.

The audit must quantify this ambiguity before a roster policy is selected.

Research outputs
1. Competition membership summary

File:

competition_membership_summary.csv

One row per competition-season.

Recommended columns:

competition
competition_type
competition_id
season_id
season_year

row_count
team_count
player_count
unique_membership_count

missing_team_id
missing_team_name
missing_player_id
missing_player_name

duplicate_membership_rows
multi_club_player_count
2. Club membership summary

File:

club_membership_summary.csv

One row per competition-season-team.

Recommended columns:

competition
competition_id
season_id
season_year

team_id
team

membership_rows
unique_players
duplicate_player_rows

multi_club_players
squad_size_flag

Suggested squad_size_flag values:

small
plausible
large
3. Multiple-club membership audit

File:

multi_club_memberships.csv

One row per player with more than one team in the same competition-season.

Recommended columns:

competition
competition_id
season_id
season_year

player_id
player

team_count
team_ids
teams
membership_rows

This file will later inform transfer-handling policy.

4. Duplicate membership audit

File:

duplicate_memberships.csv

Contains every duplicated canonical membership key.

Recommended columns:

competition_id
season_id
team_id
player_id
duplicate_count
5. Identifier consistency audit

File:

identifier_consistency.csv

Records cases where:

team_id → multiple team names

or:

player_id → multiple player names

Recommended columns:

entity_type
competition_id
season_id
entity_id
name_count
names
6. Study metadata

File:

study_metadata.json

Recommended content:

study_id
input_path
input_row_count
competition_season_count
generated_at
small_squad_threshold
large_squad_threshold
output_files
7. Human-readable report

File:

STUDY_046_RESULTS.md

The report should summarize:

competitions and seasons covered;
number of clubs;
number of unique players;
squad-size distribution;
duplicate memberships;
multiple-club players;
identifier inconsistencies;
whether season-level membership is sufficient for Version 1 club rosters.
Statistical summaries

Across club-season records, calculate:

minimum unique players
maximum unique players
mean unique players
median unique players
standard deviation
first quartile
third quartile

Across competition-seasons, calculate:

teams per season
players per season
multi-club players per season
missing-value rates
duplicate rates
Transfer interpretation policy

A player belonging to multiple clubs in the same competition-season must not be
silently assigned to one club.

Study 046 only identifies these cases.

A later implementation must choose among policies such as:

include player in every observed club-season membership
use the club associated with the latest evidence
use match-date membership
use lineup-level evidence
split the season into validity intervals
exclude ambiguous records

The correct policy should depend on the intended prediction timestamp.

For season-level descriptive representations, multiple observed memberships may
be acceptable.

For pre-match historical prediction, date-aware membership will eventually be
required.

Success criteria

Study 046 supports implementation of a Version 1 Competition Player Repository
when all of the following are true:

Competition-season identifiers are present and stable.
Team and player IDs are present for nearly all rows.
Exact duplicate memberships are absent or safely removable.
Team-ID and player-ID naming conflicts are rare and explainable.
Most club-season player counts are football-plausible.
Multiple-club memberships are measurable and represent a manageable minority.
A club-season roster can be constructed without using profile current_team.
Known temporal limitations are documented.
Possible conclusions
Conclusion A — Season-level membership is sufficient

If the data is clean and multi-club ambiguity is limited:

Proceed to CompetitionPlayerRepository Version 1.

Version 1 may use:

competition_id
season_id
team_id

as the roster lookup key.

Conclusion B — Season-level membership requires safeguards

If transfers and duplicates are common but manageable:

Proceed with Version 1 plus explicit ambiguity flags.

The repository may return membership metadata alongside player IDs.

Conclusion C — Date-aware membership is required first

If many players are associated with multiple teams and no reliable season-level
assignment can be made:

Do not build production club rosters yet.

Instead, acquire:

match lineups;
event dates;
appearance-by-team evidence;
transfer validity intervals.
Proposed implementation

The audit script will be created at:

research/study_046_club_membership/
audit_competition_player_membership.py

It will:

load sofascore_players.csv;
validate required columns;
normalize identifiers without changing source identity;
compute competition-season summaries;
compute club-season summaries;
detect exact duplicate memberships;
detect multiple-club player memberships;
audit ID-to-name consistency;
write all CSV and metadata outputs;
print a concise terminal summary.
Decision gate

No Competition Player Repository or Club Roster Builder should be implemented
until Study 046 has been run and interpreted.

The study is complete only after the results are reviewed and one of the three
conclusions above is selected.