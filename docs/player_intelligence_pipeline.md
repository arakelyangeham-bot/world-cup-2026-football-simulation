player_intelligence_pipeline.md

# Player Intelligence Pipeline

## Purpose

The Player Intelligence pipeline converts historical Sofascore
competition-season player evidence into canonical player-level
attributes and role ratings.

The pipeline deliberately separates:

1. player identity,
2. historical statistical evidence,
3. feature engineering,
4. historical evidence weighting,
5. player-level statistical representation,
6. attribute construction,
7. role-rating construction.

This separation prevents player identity and competition-season
provenance from being destroyed before historical evidence has been
weighted.

---

## Production Pipeline

The canonical production runner is:

```powershell
python -m scripts.run_player_intelligence_pipeline

It executes six stages:

scripts.resolve_player_evidence
scripts.sofascore_feature_engineering
scripts.build_canonical_player_registry
scripts.build_weighted_player_features
scripts.score_player_attributes
scripts.build_player_ratings_v4
Architecture
Evidence branch
data/raw/sofascore/sofascore_player_stats.csv
        |
        v
resolve_player_evidence.py
        |
        v
data/processed/canonical_player_evidence.csv
        |
        v
sofascore_feature_engineering.py
        |
        v
data/processed/canonical_player_features.csv
        |
        v
build_weighted_player_features.py
        |
        v
data/processed/wc_2026_model_features.csv
Identity branch
data/raw/sofascore/sofascore_player_profiles.csv
        |
        v
build_canonical_player_registry.py
        |
        v
data/processed/canonical_player_registry.csv
Player representation and ratings
wc_2026_model_features.csv
        |
        v
score_player_attributes.py
        |
        v
player_attribute_scores.csv
        |
        v
build_player_ratings_v4.py
        |
        v
player_ratings.csv

The canonical registry is used as the authoritative player identity
and role metadata source for downstream rating construction.

Stage Responsibilities
1. resolve_player_evidence.py

Grain:

competition_id × season_id × canonical_player_id

Responsibilities:

preserve original source player_id;
apply reviewed player-ID aliases;
create canonical_player_id;
detect canonical evidence collisions;
verify that colliding evidence is equivalent;
require a native canonical source row for equivalent collisions;
deterministically retain the native canonical observation;
prevent duplicate canonical competition-season task keys.

This stage must not aggregate historical evidence across competitions
or seasons.

2. sofascore_feature_engineering.py

Grain:

competition_id × season_id × canonical_player_id

Responsibilities:

preserve the incoming evidence population;
calculate per-90 player features;
retain competition-season provenance.

This stage does not calculate final player evidence confidence and
does not aggregate historical evidence.

3. build_canonical_player_registry.py

Grain:

canonical_player_id

Source:

sofascore_player_profiles.csv

Responsibilities:

provide one authoritative current profile row per canonical player;
preserve nationality and current-team metadata;
map Sofascore detailed positions into project role eligibility;
derive the project's primary role;
guarantee unique canonical player IDs.

Historical statistical evidence is not used to determine canonical
profile identity.

4. build_weighted_player_features.py

Input grain:

competition_id × season_id × canonical_player_id

Output grain:

canonical_player_id

Responsibilities:

apply competition importance;
apply season recency;
weight evidence by minutes played;
respect competition-feature availability;
aggregate raw features using only valid evidence for each feature;
calculate total weighted evidence;
calculate player-level evidence confidence;
retain evidence provenance such as competition and season counts.

Conceptually:

competition importance
    × recency
    × minutes
    × feature availability
        |
        v
weighted raw canonical-player representation

Raw features are aggregated before feature normalization.

5. score_player_attributes.py

Grain:

canonical_player_id

Responsibilities:

apply the selected feature transformation at player grain;
use robust_zscore as the production default;
combine transformed features according to
feature_attribute_manifest.csv;
produce Player Intelligence attributes.

Historical weighting does not occur in this stage.

The ordering is intentionally:

historical evidence weighting
        |
        v
raw player-level feature aggregation
        |
        v
robust_zscore at player grain
        |
        v
attribute construction
6. build_player_ratings_v4.py

Grain:

canonical_player_id

Responsibilities:

combine Player Intelligence attributes into role-specific ratings;
use canonical registry role eligibility;
preserve canonical player identity and profile metadata.
Canonical Identity

Historical Sofascore evidence may contain more than one provider
player_id for the same footballer.

Therefore:

player_id

means the source/provider identity attached to an observation, while:

canonical_player_id

means the footballer identity used by the Player Intelligence system.

Source identity must not be overwritten during canonicalization.

Reviewed mappings are stored separately and applied before historical
evidence aggregation.

Historical Aggregation Ordering

The production ordering is:

competition-season evidence
        |
        v
canonical identity resolution
        |
        v
per-90 feature engineering
        |
        v
competition / recency / availability weighting
        |
        v
raw canonical-player feature aggregation
        |
        v
robust player-grain transformation
        |
        v
attributes
        |
        v
role ratings

Do not move robust normalization before historical aggregation without
a separate research validation.

Legacy Components

The following artifacts remain useful for historical or other project
work but are not part of the modern Player Intelligence production
path:

aggregate_player_history.py
aggregate_player_history_v2.py
wc_2026_player_dataset.csv
build_player_registry.py
run_sofascore_pipeline.py

They should not be used to infer the current Player Intelligence run
order.

Production Invariants

A successful Player Intelligence build should satisfy:

no duplicate canonical competition-season evidence keys;
feature engineering preserves evidence task keys;
weighted player features contain one row per evidenced canonical
player;
attributes contain exactly the weighted-feature population;
ratings contain exactly the attribute population;
every rated canonical player exists in the canonical registry;
the canonical registry may contain players without current
statistical evidence.

The registry is therefore allowed to be a superset of the currently
rated population.

Study 101F Validation

Study 101F established the current historical-evidence architecture.

The final research comparison showed that correcting historical
competition-season weighting while preserving player-grain robust
normalization produced:

Population	Mean absolute role-rating change	Median	P90
Single-scope	0.006782	0.002563	0.016000
Multi-scope	0.128770	0.062409	0.316802

The substantially larger movement among multi-scope players supports
the interpretation that the correction primarily affected players
whose historical evidence actually required multi-scope weighting.

Production promotion tests reproduced the validated Study 101F
weighted-feature, attribute, and rating artifacts with zero numerical
difference.

The final production replay also passed all population and uniqueness
invariants.