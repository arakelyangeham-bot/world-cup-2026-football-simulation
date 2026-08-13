README.md

# League-Season Repository

## Purpose

The League-Season Repository is a canonical research dataset
containing one aggregate football-environment observation per
domestic league and season.

It was initially produced by Study 042:

**Cross-League Opta Prior Calibration**

The repository separates reusable research data from the
individual analyses that produce or consume it.

## Primary Key

Each row is uniquely identified by:

- `competition_key`
- `season_start_year`

## Current Coverage

The initial repository contains:

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

for the following season start years:

- 2023
- 2024

A season start year of `2023` represents the 2023–24 season.

## Source Data

The repository is built from Study 042 fingerprint outputs:

```text
research/studies/
└── study_042_cross_league_opta_prior_calibration/
    └── outputs/
        ├── league_fingerprints_2023.csv
        └── league_fingerprints_2024.csv

The fingerprint datasets are themselves derived from canonical
completed domestic-league match datasets.

Main Metrics

The repository includes:

matches and clubs
total goals
goals per match
home and away goals per match
mean home goal difference
home-win, draw, and away-win rates
both-teams-to-score rate
clean-sheet rates
total-goal distribution
one-goal-margin rate
three-plus-goal-margin rate
home and away points per match
Provenance Columns

Each row includes:

study_id
dataset_name
source_file
repository_created_utc

These fields identify the research workflow and source artifact
from which the row was produced.

Outputs
league_season_repository.csv
metadata.json
README.md
Update Procedure
Build and validate the canonical match dataset for a new season.
Run the Study 042 league fingerprint builder for that season.
Confirm that a new file named
league_fingerprints_<year>.csv exists.
Rerun the league-season repository builder.
Review the repository metadata and validation summary.

The builder automatically discovers all matching fingerprint files,
so no Python source changes should be required when a new season is
added.

Intended Consumers

The repository may support:

temporal league-stability analysis
home-advantage research
scoring-environment research
league clustering
environmental prior estimation
future football-model calibration studies

The repository describes league match environments. It should not
be treated as a direct ranking of league quality or strength.