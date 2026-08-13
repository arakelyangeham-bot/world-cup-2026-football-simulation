EXPECTED_GOALS_MODEL_INTERFACE

# Expected Goals Model Interface

## Status

Design specification — Version 1

## Purpose

The Expected Goals Model Interface defines the boundary between:

1. football-intelligence systems that estimate team ability;
2. feature-construction systems that combine teams with match context;
3. prediction models that estimate expected goals.

The interface exists to prevent expected-goals models from depending directly
on repository-specific dictionaries, player pipelines, or tournament code.

The intended flow is:

```text
Home Team Representation
+
Away Team Representation
+
Match Context
        ↓
Expected-Goals Feature Builder
        ↓
Expected-Goals Feature Vector
        ↓
Expected-Goals Model
        ↓
lambda_home, lambda_away

The expected-goals model should not need to know:

how player ratings were collected;
how team representations were aggregated;
which repository file supplied the team;
whether the team is a club or national team;
how scorelines are sampled after expected goals are produced.
Core principle

The expected-goals layer consumes only pre-match information.

Every feature must satisfy:

Could this value have been known at the model's prediction timestamp?

Features derived from the target match itself are forbidden.

Historical features are permitted only when generated using matches completed
before the target match.

Architectural layers
Team representation

A Team Representation describes intrinsic player-derived team ability.

Examples:

attack
midfield
defense
goalkeeper

attack_depth
midfield_depth
defense_depth

squad_quality

The representation must remain portable across competitions, venues, and
opponents.

It must not contain:

home advantage
competition environment
opponent identity
match-specific expected goals
target-match results
External strength prior

A team may have an external strength prior.

Examples:

FIFA ranking points
Elo rating
club power rating
market-value prior
dynamic form rating

The canonical model-facing field is:

rating_prior

The source-specific name, such as fifa_points, may be retained as metadata or
as a temporary compatibility alias.

External priors are not player-derived team-representation fields.

Match context

Match Context describes conditions surrounding a particular fixture.

Candidate fields include:

match_id
prediction_timestamp

competition_key
season_start_year
match_stage

home_team_designation
neutral_site

home_advantage
venue
host_country

environment_pc1
environment_pc2
competition_scoring_baseline

Context must remain separate from intrinsic team representation.

Feature construction

The feature builder combines:

home representation
away representation
home prior
away prior
match context

into one model-specific feature vector.

The feature builder owns:

field selection;
home-away differencing;
interaction terms;
missing-value policy;
feature naming;
input ordering;
validation;
leakage checks.

The prediction model should receive a completed feature vector rather than
constructing features internally from raw repositories.

Feature taxonomy

Every candidate predictor should be assigned to one of four categories.

Mandatory stable features

These are established production inputs required by the current expected-goals
pipeline.

Initial Version 1 candidates:

home_poisson_attack
away_poisson_attack

home_poisson_defense
away_poisson_defense

rating_prior_diff

The final mandatory list must be confirmed against the current production goal
model before implementation.

Optional stable features

These are conceptually valid pre-match features, but a model is not required to
use them.

Examples:

home_attack
away_attack

home_midfield
away_midfield

home_defense
away_defense

home_goalkeeper
away_goalkeeper

home_squad_quality
away_squad_quality

home_attack_depth
away_attack_depth

home_midfield_depth
away_midfield_depth

home_defense_depth
away_defense_depth

Optional features should enter production only after controlled out-of-sample
evaluation.

Experimental features

These are valid research candidates whose predictive value is not yet
established.

Examples:

environment_pc1
environment_pc2

lineup_confidence
availability_confidence
representation_evidence_score

formation indicators
squad-depth interactions
team-environment interactions

Experimental features must be introduced through explicit feature-ablation
experiments.

They should not silently enter the production model because they exist in a
repository.

Forbidden features

The following must never enter a pre-match expected-goals model for the target
fixture:

home_goals
away_goals
match_result

target-match shots
target-match possession
target-match expected goals
target-match player ratings

final competition standings
end-of-season statistics unavailable at prediction time

future lineup confirmations
future injuries or suspensions

features calculated using the target match
features calculated using later matches

A feature may be football-relevant and still be forbidden because of temporal
leakage.

Canonical feature names

Version 1 should use model-facing names independent of source datasets.

Recommended canonical names:

home_attack
away_attack

home_midfield
away_midfield

home_defense
away_defense

home_goalkeeper
away_goalkeeper

home_poisson_attack
away_poisson_attack

home_poisson_defense
away_poisson_defense

home_rating_prior
away_rating_prior
rating_prior_diff

neutral_site
home_advantage

Source names such as:

att_composite
def_composite
poisson_attack_adj
fifa_points

should be converted before the feature vector reaches the model.

Difference and interaction features

The feature builder may derive comparison features.

Examples:

attack_diff
midfield_diff
defense_diff
goalkeeper_diff
rating_prior_diff

Potential matchup interactions include:

home_attack_minus_away_defense
away_attack_minus_home_defense

home_midfield_minus_away_midfield

home_attack_depth_minus_away_defense_depth
away_attack_depth_minus_home_defense_depth

Interaction features are model-specific and should not be stored inside the
underlying Team Representation.

Expected-goals output contract

Every expected-goals model must return:

lambda_home
lambda_away

Both must be:

finite
non-negative

The model may additionally return diagnostic information:

model_name
model_version
feature_schema_version
prediction_timestamp
raw_linear_predictor_home
raw_linear_predictor_away
warnings

The scoreline engine should require only:

lambda_home
lambda_away
Conceptual Python interfaces
Match context
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MatchContext:
    match_id: str
    prediction_timestamp: datetime

    competition_key: str | None = None
    season_start_year: int | None = None
    match_stage: str | None = None

    neutral_site: bool = False
    home_advantage: float | None = None

    environment_pc1: float | None = None
    environment_pc2: float | None = None
Feature vector
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedGoalsFeatures:
    home_poisson_attack: float
    away_poisson_attack: float

    home_poisson_defense: float
    away_poisson_defense: float

    rating_prior_diff: float

    neutral_site: bool

    optional_features: dict[str, float] | None = None

    schema_version: str = "1.0"
Prediction result
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedGoalsPrediction:
    lambda_home: float
    lambda_away: float

    model_name: str
    model_version: str
    feature_schema_version: str

These classes are conceptual only.

No production implementation should be introduced until the existing goal-model
and feature-construction code has been mapped against this specification.

Feature-builder responsibilities

An Expected-Goals Feature Builder must:

accept one home-team representation;
accept one away-team representation;
accept their external priors;
accept match context;
validate all required values;
apply a documented missing-value policy;
generate canonical feature names;
ensure deterministic feature ordering;
reject temporally invalid inputs;
return a complete model-specific feature vector.

It must not:

fit model parameters;
sample scorelines;
simulate matches;
access future match information;
mutate the underlying team representations.
Model responsibilities

An Expected-Goals Model must:

accept a validated feature vector;
calculate home expected goals;
calculate away expected goals;
return finite non-negative values;
expose model and schema version information.

It should not:

query repositories directly;
reconstruct player-level information;
select lineups;
calculate competition environments;
generate scorelines.
Repository responsibilities

A team repository is a feature store.

It may expose:

team representation fields
model-facing Poisson projections
external strength priors
representation provenance
reliability metadata

Repository fields should not automatically become predictive features.

The feature builder explicitly decides which fields enter each model.

This protects the production model from accidental feature expansion.

Missing-value policy

Each feature must have one documented policy.

Permitted policies include:

reject prediction
use a neutral default
use a competition mean
use a team-level fallback
use an external prior
emit a missingness indicator

Silent coercion is prohibited.

For example, converting a missing team strength to 1.0 may be valid for a
legacy compatibility path, but the fallback must be visible in validation or
prediction metadata.

Representation provenance

These fields describe how a team representation was constructed:

representation_type
aggregation_profile
player_count
available_player_count
evidence_score

They are not automatically predictive inputs.

Their initial purposes are:

reproducibility;
validation;
filtering;
uncertainty analysis;
later feature-ablation experiments.

For example, player_count should not enter a production expected-goals model
merely because it is available.

National-team and club compatibility

The expected-goals interface should support both national teams and clubs.

The upstream construction processes may differ:

National team:
player intelligence
+
national-team prior
+
tournament context

Club:
player intelligence
+
club-strength prior
+
league context

Both should eventually produce compatible model-facing inputs.

National-team and club matches should not be combined into one training
population without an explicit statistical justification.

A shared interface does not imply a shared fitted model.

Feature schema versioning

Every feature vector should carry a schema version.

Example:

expected_goals_features_v1

A schema version changes when:

a required feature is added;
a required feature is removed;
a feature changes meaning;
a feature changes scale;
feature ordering changes for models that depend on ordering.

Adding unused repository metadata does not require a feature-schema change.

Initial model families

The interface should support several model families.

Examples:

Baseline Poisson goal model

Poisson model with rating prior

Player-derived dimensional Poisson model

Environment-augmented goal model

Machine-learning expected-goals model

Each model family may define its own feature builder while consuming the same
underlying representations and match context.

Validation requirements

Before a feature builder enters production, validate:

required fields exist
values are finite
expected scales are respected
feature ordering is deterministic
home-away orientation is correct
missing values follow policy
target leakage is absent
historical joins are time-safe

Before an expected-goals model enters production, validate:

lambda_home >= 0
lambda_away >= 0

predictions are finite
identical inputs produce identical outputs
home-away reversal behaves as expected
legacy-equivalent inputs reproduce legacy outputs
Compatibility with the current simulator

The existing simulator should not be rewritten immediately.

The first implementation should be an adapter around the current production
path.

Current team repository dictionaries
        ↓
Legacy Expected-Goals Feature Adapter
        ↓
Current goal model
        ↓
lambda_home, lambda_away

The adapter should reproduce existing expected-goals values exactly.

Only after equivalence is proven should downstream consumers migrate toward the
new interface.

Initial acceptance criteria

Version 1 is ready for implementation when:

the current production goal model's inputs are mapped explicitly;
the existing home-away feature orientation is documented;
all current missing-value fallbacks are documented;
a legacy feature adapter can reproduce current feature values;
the adapted expected-goals path reproduces current lambdas exactly;
the scoreline engine remains unchanged;
no new feature enters production without evaluation.
Non-goals

Version 1 does not:

change expected-goals coefficients;
add environmental PCA features to production;
retrain the goal model;
merge club and international training data;
redesign the scoreline engine;
change tournament simulation;
require every repository field to become a model input;
define the final club prediction model.

Its purpose is to establish a controlled boundary between representations and
prediction.