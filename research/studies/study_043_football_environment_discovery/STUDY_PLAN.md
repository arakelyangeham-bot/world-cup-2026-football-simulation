STUDY_PLAN

# Study 043 — Football Environment Discovery

## Status

Planned

## Objective

Determine whether domestic league-seasons naturally organize into
distinct football environments when represented only by measurable
match characteristics.

The study does not attempt to rank leagues by quality or strength.

Instead, it investigates whether league-seasons occupy recognizable
positions within a multidimensional football-environment space.

## Background

Study 042 established that domestic leagues differ along several
measurable dimensions, including:

- scoring intensity
- both-teams-to-score frequency
- draw frequency
- scoreline dispersion
- one-goal match frequency
- home and away scoring

Study 042 also found that some metrics were more temporally stable
than others.

Scoring-related metrics showed greater stability than league-specific
home-advantage metrics.

Study 043 therefore uses the more stable environmental characteristics
as its initial feature set.

## Research Questions

### Primary Question

Do league-seasons naturally form distinct groups when represented by
stable match-environment metrics?

### Secondary Questions

1. What are the dominant dimensions of variation among league-seasons?
2. Do observations from the same league remain close across seasons?
3. Which leagues or seasons occupy similar football environments?
4. Are any apparent clusters robust across different clustering methods?
5. Can the discovered environments be interpreted in football terms?
6. Are the environments stable enough to justify later model experiments?

## Dataset

The study uses the canonical League-Season Repository:

```text
research/datasets/league_season_repository/
└── league_season_repository.csv

Current Coverage
5 domestic leagues
2 seasons per league
10 league-season observations

The primary key is:

competition_key
season_start_year
Important Sample-Size Limitation

The current repository contains only 10 observations.

This is too small for strong or definitive clustering claims.

Therefore:

all clustering results must be treated as exploratory;
cluster-quality statistics must not be overinterpreted;
a cluster must not be treated as a permanent football archetype;
results should be used primarily to guide future data collection;
additional leagues and seasons will be needed before production use.

The first version of Study 043 is therefore an environment-mapping
experiment rather than a final clustering model.

Candidate Features

The initial feature set is:

goals_per_match
home_goals_per_match
away_goals_per_match
both_teams_to_score_rate
draw_rate
one_goal_margin_rate
three_plus_goal_margin_rate

These features describe:

total scoring intensity;
home and away scoring contributions;
likelihood that both teams score;
draw tendency;
close-match frequency;
large-margin-result frequency.
Excluded Features

The following metrics are excluded from the initial experiment:

mean_home_goal_difference
home_win_rate
home_points_per_match
away_points_per_match

Study 042 found that league-specific home-advantage measurements were
comparatively unstable between 2023–24 and 2024–25.

These metrics may be reconsidered in later sensitivity experiments.

Feature Redundancy Warning

Several selected features are related.

For example:

goals_per_match
=
home_goals_per_match
+
away_goals_per_match

Including all three may overweight scoring intensity.

The study will therefore run at least two feature specifications.

Feature Set A — Full Environmental Profile
goals per match
home goals per match
away goals per match
BTTS rate
draw rate
one-goal-margin rate
three-plus-goal-margin rate
Feature Set B — Reduced Profile
goals per match
BTTS rate
draw rate
one-goal-margin rate
three-plus-goal-margin rate

Feature Set B removes home and away scoring components to reduce
deterministic redundancy.

Preprocessing

All selected features will be standardized before PCA or clustering.

For each feature:

standardized value
=
(value - feature mean)
/
feature standard deviation

Standardization is required because the features use different scales.

The fitted scaler parameters must be saved as an output artifact.

Experimental Sequence
Phase 1 — Dataset Audit

Validate:

repository existence;
expected columns;
unique league-season primary keys;
numeric feature values;
absence of missing values;
rate values within [0, 1];
number of observations and features.
Phase 2 — Feature Scaling

Standardize the selected features.

Produce:

standardized_environment_features.csv
feature_scaler_parameters.csv
Phase 3 — Principal Component Analysis

Run PCA on both feature specifications.

Produce:

principal-component coordinates;
explained-variance ratios;
cumulative explained variance;
feature loadings.

PCA is used to answer:

Which combinations of football metrics explain the largest variation
among league-seasons?

Phase 4 — PCA Interpretation

Interpret each principal component using its feature loadings.

Possible interpretations may include:

scoring intensity;
match openness;
draw or compactness tendency;
close-match versus blowout tendency.

These interpretations must be based on the observed loadings rather
than assigned in advance.

Phase 5 — Hierarchical Clustering

Run hierarchical clustering on standardized features.

Initially evaluate:

Ward linkage;
Euclidean distance.

Produce:

linkage structure;
dendrogram-ready data;
candidate cluster assignments.

Because the sample is small, hierarchical clustering will be treated
as the primary exploratory clustering method.

Phase 6 — K-Means Comparison

Run K-means only as a comparison method.

Evaluate a small cluster range such as:

k = 2
k = 3
k = 4

Use multiple random initializations and a fixed random seed.

Produce:

cluster assignments;
standardized cluster centroids;
inertia;
silhouette score where mathematically valid.
Phase 7 — Gaussian Mixture Comparison

Run Gaussian mixture models only if the sample size permits stable
estimation.

Compare a small number of components.

If covariance estimation is unstable, record that limitation rather
than forcing a result.

Phase 8 — Cross-Method Agreement

Compare whether hierarchical clustering, K-means, and Gaussian
mixtures place the same league-seasons together.

Possible conclusions:

strong agreement;
partial agreement;
weak or unstable grouping.

A cluster should not be interpreted as meaningful merely because one
algorithm produced it.

Phase 9 — Temporal Consistency

For every league, compare the positions of its 2023 and 2024
observations.

Measure:

standardized Euclidean distance;
PCA-space distance;
whether both seasons receive the same cluster assignment.

This phase asks:

Is a league more similar to itself across seasons than to other
leagues?

Phase 10 — Football Interpretation

Describe any recurring groups in football terms.

Example descriptions might include:

high-scoring and open;
low-scoring and draw-prone;
high-BTTS but relatively balanced;
low-BTTS with fewer large-margin results.

These labels must be derived from cluster profiles and must not imply
league quality.

Validation Principles

The study follows these rules:

No league labels are used as model inputs.
No cluster is called meaningful solely from visual appearance.
Results must be compared across feature specifications.
Results must be compared across clustering algorithms.
Small-sample limitations must remain explicit.
No production model change follows directly from this study.
Environment discovery and predictive usefulness are separate
research questions.
Planned Outputs
research/studies/study_043_football_environment_discovery/
├── STUDY_PLAN.md
├── outputs/
│   ├── environment_feature_audit.csv
│   ├── standardized_environment_features_full.csv
│   ├── standardized_environment_features_reduced.csv
│   ├── feature_scaler_parameters_full.csv
│   ├── feature_scaler_parameters_reduced.csv
│   ├── pca_coordinates_full.csv
│   ├── pca_coordinates_reduced.csv
│   ├── pca_explained_variance_full.csv
│   ├── pca_explained_variance_reduced.csv
│   ├── pca_loadings_full.csv
│   ├── pca_loadings_reduced.csv
│   ├── hierarchical_cluster_assignments.csv
│   ├── kmeans_cluster_assignments.csv
│   ├── clustering_evaluation.csv
│   └── temporal_environment_distances.csv
└── reports/
    └── STUDY_043_REPORT.md
Reproducibility

All stochastic procedures will use a fixed random seed:

42

Software versions and parameter choices should be recorded in the
final report.

Success Criteria

Study 043 will be considered successful if it determines:

the dominant axes of variation among league-seasons;
whether league-seasons show recognizable environmental similarity;
whether groupings are consistent across feature sets;
whether groupings are consistent across algorithms;
whether a league remains near itself across seasons;
whether the available sample is sufficient for meaningful
clustering conclusions.

A valid conclusion may be:

The current dataset is too small to support stable clustering.

That outcome would still be scientifically useful.

Production Boundary

Study 043 will not directly modify:

player priors;
team-strength priors;
expected-goal generation;
scoreline sampling;
the production match engine;
the World Cup simulator.

Any production integration would require a later supervised validation
study demonstrating predictive improvement on held-out matches.

Initial Hypothesis

League-seasons will vary primarily along a scoring-openness dimension,
with goals per match, BTTS rate, and large-margin frequency contributing
strongly to the first principal component.

This is a hypothesis to test, not an assumed result.