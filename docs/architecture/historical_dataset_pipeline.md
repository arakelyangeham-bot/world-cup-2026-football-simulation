historical_dataset_pipeline.md

# Historical Domestic-League Dataset Pipeline

## Status

**Subsystem version:** 1.0  
**Lifecycle status:** Production-stable for the current domestic-league research workflow

This document defines the architecture, validation policy, outputs, and known limitations of the historical domestic-league dataset pipeline.

The subsystem converts raw Sofascore season data into canonical match datasets suitable for downstream football research.

---

# 1. Purpose

The historical domestic-league dataset pipeline exists to transform raw competition-season event data into:

- structurally validated league datasets;
- reproducible canonical match records;
- explicit records of excluded and incomplete events;
- team-level schedule summaries;
- machine-readable dataset metadata;
- documented source anomalies when the raw source cannot satisfy the official competition structure.

The pipeline is intentionally strict.

A dataset is not accepted merely because it contains plausible scores or approximately the expected number of matches. It must satisfy schema, event, score, fixture, and league-structure requirements.

When the source data contains a verified and precisely documented limitation, the dataset may be accepted only through the known source-anomaly policy.

---

# 2. Architectural Scope

The subsystem currently supports registered domestic leagues whose regular seasons follow a double round-robin structure.

For each pair of clubs:

- each club hosts the other once;
- every unordered club pairing occurs twice;
- every directed home-away fixture occurs exactly once.

Competition-specific structural expectations are provided by the domestic-league rules layer.

These expectations include:

- number of clubs;
- completed regular-season match count;
- matches per club;
- home matches per club;
- away matches per club;
- number of unique club pairings.

League rules may vary by season when a competition changes format.

---

# 3. Primary Components

## 3.1 Competition Registry

The competition registry provides canonical competition metadata, including:

- competition key;
- display name;
- category;
- filename pattern;
- source-specific identifiers.

The historical builder accepts only competitions registered with the category:

```text
domestic_league

3.2 Sofascore Season Registry

The Sofascore season registry maps a registered competition and season start year to the corresponding source identifiers.

A season start year represents the first calendar year of the season.

Examples:

2021 -> 2021–22
2024 -> 2024–25

The registry is validated independently before season data is processed.

3.3 Domestic-League Rules

The domestic-league rules layer defines the official expected structure of a league-season.

Typical rules include:

team_count
matches_per_team
home_matches_per_team
away_matches_per_team
completed_match_count
unique_pairing_count

These rules represent the official competition structure.

They must not be weakened to accommodate incomplete source data.

Source limitations are handled separately through the known source-anomaly registry.

3.4 Historical Dataset Builder

Primary module:

research/datasets/domestic_leagues/
    validate_and_build_historical_matches.py

This module orchestrates:

raw dataset loading;
schema validation;
event classification;
source-status filtering;
regular-season filtering;
score validation;
fixture-integrity analysis;
source-anomaly resolution;
official league-structure validation;
canonical dataset construction;
output generation.

The builder is the primary entry point for producing canonical historical domestic-league datasets.

3.5 Fixture-Integrity Analyzer

Module:

research/datasets/domestic_leagues/
    fixture_integrity.py

The fixture-integrity analyzer compares the completed regular-season dataset against the full expected directed schedule.

It reports:

expected directed fixtures;
observed match rows;
observed unique fixtures;
missing fixtures;
unexpected fixtures;
duplicate fixtures.

For an 18-team double round-robin league:

18 x 17 = 306 directed fixtures

For a 20-team double round-robin league:

20 x 19 = 380 directed fixtures

Fixture integrity is evaluated before aggregate structural validation so that failures can identify the exact affected fixture rather than only reporting a count discrepancy.

3.6 Known Source-Anomaly Registry

Module:

research/datasets/domestic_leagues/
    known_source_anomalies.py

The known source-anomaly registry records verified discrepancies between:

the official expected league schedule; and
the valid completed regular-season events available in the raw source dataset.

An anomaly entry includes:

competition key;
season start year;
exact missing directed fixture or fixtures;
associated source event IDs;
human-readable explanation.

A registered anomaly does not broadly relax validation.

The observed dataset must match the registered anomaly exactly.

The anomaly mechanism rejects datasets containing:

additional missing fixtures;
unexpected fixtures;
duplicate fixtures;
a completed-match count inconsistent with the documented anomaly.

The registry therefore acts as a narrowly defined, reproducible source-integrity record rather than a general exception list.

3.7 Fixture Investigation Utility

Module:

research/diagnostics/
    investigate_fixture_events.py

The investigation utility is separate from the canonical builder.

Its responsibility is to explain why a fixture fails validation by searching the raw season dataset for:

a club pairing in either home-away orientation; or
a specific source event ID.

It displays available source information such as:

event ID;
date;
home and away clubs;
score;
status code;
status description;
stage;
round;
round number.

The diagnostic utility answers:

Why is this fixture missing from the canonical completed dataset?

The canonical builder answers:

Is this dataset valid and acceptable?

These responsibilities remain separate.

4. Processing Pipeline
Raw Sofascore season CSV
        |
        v
Competition and season resolution
        |
        v
Raw schema validation
        |
        v
Event ID normalization and uniqueness validation
        |
        v
Score-based row classification
        |
        +-----------------------------+
        |                             |
        v                             v
Score-complete events          Incomplete events
        |
        v
Official completion-status filtering
        |
        +-----------------------------+
        |                             |
        v                             v
Officially completed           Excluded non-final
events                         scored events
        |
        v
Regular-season stage filtering
        |
        +-----------------------------+
        |                             |
        v                             v
Completed regular-season       Excluded completed
matches                        non-regular-stage events
        |
        v
Required-value and score validation
        |
        v
Duplicate completed-fixture validation
        |
        v
Team match-summary construction
        |
        v
Fixture-integrity analysis
        |
        +-------------------------------------------+
        |                                           |
        v                                           v
Perfect fixture structure                 Fixture-integrity failure
        |                                           |
        v                                           v
Strict league-structure                   Known source-anomaly lookup
validation                                          |
        |                              +------------+-------------+
        |                              |                          |
        |                              v                          v
        |                    Exact registered match       No exact registered match
        |                              |                          |
        |                              v                          v
        |                    Accept with explicit              FAIL
        |                    anomaly metadata
        |
        v
Canonical dataset construction
        |
        v
Output and metadata generation
5. Raw Dataset Requirements

The source CSV must contain the required fields expected by the builder.

The current required schema includes:

event_id
date
stage
round
round_number
home_team
home_team_id
away_team
away_team_id
home_score
away_score
status_code
status_desc
winner

The dataset must not be empty.

Event IDs must be numeric and unique within the raw season dataset.

Dates are parsed as UTC timestamps.

Scores are converted to numeric values.

6. Event Classification
6.1 Score-Complete Events

An event is initially considered score-complete when both:

home_score
away_score

are present.

Score completeness alone does not establish that the match was officially completed.

6.2 Incomplete Events

Events missing either home or away score are separated into the incomplete-event dataset.

Examples may include:

postponed fixtures;
scheduled fixtures;
cancelled fixtures;
source records without a final score.

Incomplete events do not enter the canonical completed-match dataset.

7. Official Completion Filtering

A score-complete event must also possess a recognized final status.

The current final-status policy accepts source events with:

status_code = 100

or a normalized status description of:

ended

Score-complete events with non-final statuses are excluded.

Examples include:

abandoned;
postponed;
cancelled;
suspended;
interrupted.

This distinction is essential because an abandoned event may contain a score without representing an officially completed match.

8. Regular-Season Stage Filtering

Only officially completed events belonging to the registered regular-season competition stage enter the canonical league dataset.

Stage labels are normalized before comparison by:

converting text to lowercase;
removing surrounding whitespace;
removing punctuation and non-alphanumeric separators.

This allows equivalent source labels such as:

La Liga
LaLiga

to compare consistently.

Associated events from other stages are excluded, including:

relegation or promotion playoffs;
championship rounds;
qualification stages;
other competition phases returned through the same source season.
9. Completed-Match Validation

The completed regular-season dataset is checked for:

required non-null values;
valid home and away clubs;
different home and away teams;
nonnegative scores;
integer-valued scores;
duplicate completed fixtures.

A completed fixture duplicate is investigated rather than silently removed.

10. Team Match Summary

The pipeline creates a team-level match summary containing:

team
home_matches
away_matches
total_matches

For a structurally complete season, each club must match the official season rules.

For example, in an 18-team double round-robin league:

home_matches = 17
away_matches = 17
total_matches = 34

When a season is accepted with a known source anomaly, one or more clubs may have a reduced observed count corresponding exactly to the documented missing fixture.

11. Fixture-Integrity Policy

The expected schedule is constructed from every ordered permutation of distinct registered clubs.

For teams A and B, both directed fixtures are expected:

A vs B
B vs A

The analyzer detects three defect classes.

Missing Fixtures

Expected directed fixtures absent from the completed regular-season dataset.

Unexpected Fixtures

Observed directed fixtures that are not part of the expected league schedule.

Duplicate Fixtures

Directed fixtures represented more than once in the completed regular-season dataset.

A normal season passes only when all three counts are zero and the observed match count equals the expected directed-fixture count.

12. Validation Outcomes

The pipeline supports three outcomes.

12.1 PASSED

A dataset receives:

PASSED

when it satisfies:

raw schema requirements;
event ID requirements;
score requirements;
official completion filtering;
regular-stage filtering;
fixture integrity;
official league structure;
pairing structure.

The metadata records:

"validation_status": "passed"
12.2 ACCEPTED WITH KNOWN SOURCE ANOMALY

A dataset receives:

ACCEPTED WITH KNOWN SOURCE ANOMALY

only when:

fixture integrity fails;
an anomaly exists for the exact competition and season;
the observed missing fixtures exactly equal the registered missing fixtures;
no unexpected fixtures exist;
no duplicate fixtures exist;
the observed completed-match count agrees with the documented number of missing fixtures;
the expected number of clubs is still present.

The official expected structure remains unchanged.

For example, the report may retain:

Expected completed matches: 306
Observed completed matches: 305

The discrepancy is preserved rather than hidden.

The metadata records:

"validation_status": "accepted_with_known_source_anomaly"

and includes the complete anomaly record.

12.3 FAILED

A dataset fails when any unresolved validation condition exists.

Examples include:

an unregistered missing fixture;
additional missing fixtures beyond a registered anomaly;
unexpected fixtures;
duplicate directed fixtures;
an invalid club count;
an invalid score;
duplicate event IDs;
missing required columns;
an incorrect competition stage;
an anomaly whose observed evidence does not exactly match its registry entry.

Failures must be investigated.

Validation rules must not be weakened solely to force acceptance.

13. Canonical Completed Dataset

Accepted completed regular-season events are transformed into the canonical historical-match schema.

Derived fields include:

goal_difference
total_goals
outcome
completed
competition_key
season_start_year

The match outcome is represented as one of:

home_win
draw
away_win

The canonical dataset is ordered chronologically by:

date
event_id
14. Generated Outputs

For each competition-season, the builder writes five outputs.

14.1 Completed Matches
<competition>_<year>_completed_matches.csv

Contains the canonical completed regular-season match dataset.

This is the primary downstream research input.

14.2 Incomplete Events
<competition>_<year>_incomplete_events.csv

Contains raw events without complete score information.

This file preserves source evidence for postponed, scheduled, cancelled, or otherwise incomplete records.

14.3 Excluded Completed Events
<competition>_<year>_excluded_completed_events.csv

Contains scored events excluded from the canonical dataset.

Each row includes an exclusion reason such as:

non_final_status
non_regular_season_stage

Examples include:

abandoned scored events;
relegation or promotion playoff matches;
other non-regular-stage events.
14.4 Team Match Summary
<competition>_<year>_team_match_summary.csv

Contains the number of observed home, away, and total matches for every club.

This supports structural auditing and downstream inspection.

14.5 Dataset Metadata
<competition>_<year>_dataset_metadata.json

Records:

competition key;
competition name;
season start year;
source path;
raw row count;
canonical completed-match count;
incomplete-event count;
excluded-event counts;
club count;
validation status;
canonical output path;
known source-anomaly details when applicable.

The metadata file must remain associated with the canonical dataset.

This is especially important for anomaly-accepted seasons.

15. Verified Current Source Anomalies
15.1 Bundesliga 2021–22

Competition key:

bundesliga

Season start year:

2021

Missing directed fixture:

VfL Bochum 1848 vs Borussia M'gladbach

Associated source event:

9594319

Observed source representation:

fixture appears as an abandoned event;
it contains a score;
it does not have an accepted final status;
excluding it leaves 305 completed regular-season fixtures rather than 306.

The opposite fixture:

Borussia M'gladbach vs VfL Bochum 1848

is present as an ended match.

15.2 Ligue 1 2025–26

Competition key:

ligue_1

Season start year:

2025

Missing directed fixture:

Nantes vs Toulouse

Associated source event:

14061912

Observed source representation:

fixture appears as an abandoned event;
it contains a score;
it does not have an accepted final status;
excluding it leaves 305 completed regular-season fixtures rather than 306.

The opposite fixture:

Toulouse vs Nantes

is present as an ended match.

16. Important Interpretation of Source Anomalies

The anomaly registry describes the contents of the acquired raw source dataset.

It does not automatically establish:

why the provider represents an event in that manner;
whether another provider endpoint contains a replacement record;
whether the official result was later awarded, replayed, or administratively resolved;
whether the source representation may change in the future.

Claims about official historical resolution require separate authoritative verification.

The pipeline's defensible conclusion is narrower:

The acquired raw season dataset does not contain an accepted final event for the documented directed fixture.

This distinction prevents the architecture document from overstating what the available evidence proves.

17. Downstream Research Requirements

A downstream study using canonical historical league data must:

read the accompanying dataset metadata;
preserve validation status;
identify anomaly-accepted seasons;
avoid describing an anomaly-accepted season as structurally complete;
consider the effect of missing fixtures on calculated metrics;
document whether anomaly-accepted seasons are included or excluded.

Examples of potentially affected quantities include:

goals per match;
draw rate;
both-teams-to-score rate;
scoreline frequencies;
margin distributions;
team-level averages;
uncertainty estimates.

One missing match generally creates a small aggregate effect, but it must remain visible and documented.

18. Operational Commands
Build and validate a season
python -m research.datasets.domestic_leagues.validate_and_build_historical_matches --competition <competition_key> --year <season_start_year>

Example:

python -m research.datasets.domestic_leagues.validate_and_build_historical_matches --competition bundesliga --year 2024
Investigate a club pairing
python -m research.diagnostics.investigate_fixture_events --competition <competition_key> --year <season_start_year> --home "<home_team>" --away "<away_team>"

Example:

python -m research.diagnostics.investigate_fixture_events --competition bundesliga --year 2021 --home "VfL Bochum 1848" --away "Borussia M'gladbach"
Investigate an event ID
python -m research.diagnostics.investigate_fixture_events --competition <competition_key> --year <season_start_year> --event-id <event_id>

Example:

python -m research.diagnostics.investigate_fixture_events --competition bundesliga --year 2021 --event-id 9594319
19. Expansion Procedure

For each newly added domestic league-season:

Register the competition if necessary.
Register the source season identifiers.
Validate the season registry.
Acquire the raw season dataset.
Run the historical dataset builder.
Accept the dataset if it passes.
Investigate any failure at the fixture level.
Add a source anomaly only after the discrepancy is reproduced and precisely documented.
Rerun validation.
Preserve all canonical outputs and metadata.
Add the season to the research repository only after acceptance.

Anomaly registration must never be used as a shortcut around unexplained validation failures.

20. Version 1.0 Stability Policy

The historical domestic-league dataset pipeline is considered stable at Version 1.0.

Expected future work includes:

registering additional seasons;
registering additional competitions;
adding verified source anomalies;
correcting confirmed implementation defects;
extending season-aware competition rules where formats vary.

The core architecture should not be redesigned merely because a new dataset fails validation.

A structural change is warranted only when a genuinely new class of competition or source behavior cannot be represented through the current abstractions.

The preferred operating principle is:

Expand the data while keeping the pipeline stable.

21. Scientific Principles

The subsystem follows these principles:

Official competition rules and observed source data are separate concepts.
A scored event is not necessarily an officially completed match.
Aggregate match counts are insufficient without fixture-level integrity.
Source limitations must be documented rather than hidden.
Known anomalies must match observed evidence exactly.
Canonical datasets must retain machine-readable validation metadata.
Diagnostics explain failures but do not weaken validation policy.
New data should test the architecture rather than automatically change it.
Uncertainty and incompleteness must remain visible to downstream research.
Acceptance is an evidence-based decision, not merely successful script execution.