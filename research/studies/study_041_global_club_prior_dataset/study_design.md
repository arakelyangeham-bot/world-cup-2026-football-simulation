study_design.md

# Study 041 — Global Club Prior Dataset

## Overview

The purpose of Study 041 is to build a reproducible external-strength dataset for club football.

The dataset will provide the club-football equivalent of the FIFA-ranking prior currently used for national teams.

The initial source will be the Opta Power Rankings.

The study will begin with the clubs required for the first Premier League simulation and later expand to clubs from additional domestic and continental competitions.

---

## Research and Engineering Question

How should external club-strength ratings be acquired, stored, validated, and transformed into a canonical prior that can be consumed by the Production Football Model?

---

## Motivation

The existing Production Football Model combines:

- player-derived team representations,
- Poisson attack and defense features,
- an external overall-strength prior.

For national teams, the external prior is derived from FIFA points.

Club teams do not have FIFA points.

A global club-rating source is therefore required before the Production Football Model can support domestic leagues, continental competitions, and intercontinental club tournaments.

Opta Power Rankings are a strong candidate because they provide:

- a common global scale,
- ratings rather than ordinal ranks alone,
- comparisons across leagues and countries,
- regularly updated club-strength estimates.

---

## Study Objectives

Study 041 will:

1. Acquire a reproducible snapshot of Opta club ratings.
2. Preserve the raw source values without modification.
3. Validate club names, ratings, rankings, and snapshot metadata.
4. Build a canonical project-level club prior dataset.
5. Design a calibration layer between raw Opta ratings and `rating_prior`.
6. Produce data suitable for club repository construction.
7. Document limitations and provenance.

---

## Scope

### Initial Scope

The first version will cover the clubs required for the Premier League competition definition.

### Future Scope

The dataset may later expand to support:

- La Liga
- Serie A
- Bundesliga
- Ligue 1
- UEFA Champions League
- UEFA Europa League
- UEFA Conference League
- FIFA Club World Cup
- Copa Libertadores
- MLS
- AFC club competitions
- CAF club competitions

---

## Data Layers

Study 041 will distinguish between three data layers.

### 1. Raw Snapshot

A direct representation of the source data at a particular point in time.

Suggested file:

```text
data/raw/opta_power_rankings/opta_power_rankings_YYYY_MM_DD.csv

The raw snapshot should contain only data obtained from the source, plus provenance metadata.

It should not contain calibrated or model-derived values.

2. Enriched Snapshot

The raw snapshot supplemented with project metadata such as:

canonical club name,
league,
country,
confederation.

Suggested file:

data/processed/opta_power_rankings_enriched_YYYY_MM_DD.csv
3. Canonical Global Club Prior Dataset

The project-facing dataset consumed by repository builders and experiments.

Suggested file:

data/processed/global_club_prior_dataset.csv

This dataset may include a calibrated rating_prior, but the raw Opta rating must always remain available separately.

Proposed Raw Snapshot Schema
source_rank
club_raw
opta_rating
global_rank
ranking_change_7_days
snapshot_date
source_name
source_url
retrieved_at
Field Definitions
source_rank: position displayed in the source table.
club_raw: club name exactly as shown by the source.
opta_rating: unmodified Opta Power Rating.
global_rank: global ranking position.
ranking_change_7_days: reported seven-day change where available.
snapshot_date: date represented by the ranking snapshot.
source_name: source label, such as Opta Power Rankings.
source_url: page from which the snapshot was collected.
retrieved_at: timestamp when the project acquired the data.
Proposed Canonical Dataset Schema
club
club_raw
league
country
confederation
opta_rating
global_rank
rating_prior
rating_prior_method
rating_source
snapshot_date
source_url
Important Distinction

opta_rating is a raw external observation.

rating_prior is a model input.

They must not be treated as interchangeable until a calibration method has been selected and validated.

Canonical Club Names

The canonical club field will be used as the merge key across:

competition participant lists,
club repositories,
player rosters,
fixture generation,
simulation outputs.

The raw source name must be preserved as club_raw.

Name normalization should be explicit and auditable.

Examples of possible normalization issues include:

Brighton & Hove Albion versus Brighton & Hove
Manchester United versus Man United
Wolverhampton Wanderers versus Wolves
Tottenham Hotspur versus Tottenham

A dedicated alias table may eventually be required.

Acquisition Pipeline

The intended acquisition workflow is:

Source page
    ↓
Download HTML
    ↓
Parse ranking table
    ↓
Write raw snapshot
    ↓
Validate raw snapshot
    ↓
Normalize and enrich club metadata
    ↓
Build canonical dataset

Scraping, validation, enrichment, and calibration should remain separate steps.

Scraper Requirements

The Opta snapshot scraper should:

accept the source URL as an argument or configuration value,
use a descriptive browser user agent,
apply a request timeout,
fail clearly on non-successful HTTP responses,
verify that the expected table exists,
extract all required columns,
preserve source values,
record retrieval metadata,
write a dated snapshot rather than overwrite prior snapshots,
avoid silently returning incomplete data.

The scraper should not:

calculate rating_prior,
assign leagues or countries,
alter club names,
overwrite raw ratings,
assume that table CSS class names are permanent.
Validation Requirements

The raw snapshot validator should check:

required columns exist,
no duplicate source rows exist,
club names are non-empty,
Opta ratings are numeric,
Opta ratings fall within the expected source range,
global ranks are positive integers,
global ranks are unique where appropriate,
snapshot date is present,
source URL is present,
the expected target clubs are present,
no unexpected missing values occur.

For the initial Premier League subset, validation should confirm that every competition participant has a matching Opta record.

Calibration Problem

The current calibrated expected-goals model was fitted using FIFA-point differences.

Opta Power Ratings use a different numerical scale.

Therefore, raw Opta ratings cannot automatically be substituted for FIFA points without changing the effective magnitude of the fitted coefficient.

The general feature has now been renamed:

rating_prior

but its scale remains important.

Candidate Calibration Methods
Method A — Linear Scale Mapping

Transform Opta ratings to a FIFA-like numerical range:

rating_prior = intercept + scale × opta_rating

Advantages:

simple,
transparent,
easy to reproduce.

Limitations:

arbitrary unless fitted against evidence,
may distort rating differences.
Method B — Distribution Matching

Match the mean and standard deviation of Opta ratings to the FIFA-prior distribution used during model training.

Advantages:

preserves standardized differences,
aligns input scale with the fitted model.

Limitations:

depends on the selected club and national-team populations,
does not guarantee football-equivalent meaning.
Method C — Percentile Mapping

Map club ratings and FIFA ratings through corresponding distribution percentiles.

Advantages:

less sensitive to differences in raw scale shape.

Limitations:

may produce unstable tails,
depends heavily on reference populations.
Method D — Difference-Scale Calibration

Estimate a multiplier so typical Opta rating differences produce effects comparable to typical FIFA-point differences.

Advantages:

focuses on the exact quantity used by the model.

Limitations:

still indirect,
requires a defensible reference distribution.
Method E — Club-Specific Model Retraining

Retrain the expected-goals model directly using:

historical club match results,
club attack and defense features,
Opta-rating differences.

Advantages:

scientifically strongest long-term approach,
estimates a coefficient appropriate to the Opta scale.

Limitations:

requires historical club match data,
requires additional calibration and validation work.
Initial Policy

Until calibration is completed:

raw Opta ratings will be stored as opta_rating,
any transformed rating_prior will be labeled experimental,
temporary mappings will be used only for structural validation,
no placeholder transformation will be described as production-calibrated.
Deliverables

Study 041 should eventually contain:

studies/
└── study_041_global_club_prior_dataset/
    ├── study_design.md
    ├── scrape_opta_snapshot.py
    ├── validate_opta_snapshot.py
    ├── build_global_club_prior_dataset.py
    ├── calibrate_opta_rating_prior.py
    ├── report.md
    └── README.md

Data outputs should remain under the appropriate data/ or outputs/ directories rather than being stored directly inside the study source directory.

Success Criteria

Study 041 will be considered complete when:

A dated raw Opta snapshot can be acquired reproducibly.
The snapshot passes structural validation.
Club names can be matched to competition participants.
A canonical global club prior dataset can be produced.
Raw Opta ratings remain preserved.
A documented calibration method produces rating_prior.
The resulting dataset can build a club repository.
The Premier League simulation can run without placeholder prior data.
Limitations
Opta ratings are externally maintained and may change frequently.
Source HTML structure may change.
Historical snapshots may not always be available.
The rating methodology is not controlled by this project.
Raw Opta ratings are not automatically compatible with the current national-team Poisson coefficient.
Player-derived club strengths remain a separate data requirement.
Long-Term Role

The Global Club Prior Dataset should become the common external-strength layer for club simulations.

It should support domestic leagues, continental competitions, and intercontinental tournaments without requiring competition-specific rating systems.

The intended architecture is:

Player-derived club strengths
            +
Global external club prior
            ↓
Canonical club repository
            ↓
Production Football Model
            ↓
Competition Framework

This study therefore establishes the first reusable global data asset for the club-football branch of the project.