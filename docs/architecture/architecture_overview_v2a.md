architecture_overview_v2a

# World Cup 2026 Football Simulation Platform

## Architecture Overview — Version 2A

## 1. Purpose

Version 2A establishes the project as a reusable computational
football research platform rather than a single-purpose World Cup
simulator.

The architecture supports:

- player-level football intelligence;
- competition-aware team representation;
- persisted production repositories;
- date-aware rating priors;
- frozen expected-goal model artifacts;
- deterministic fixture prediction;
- scoreline-first stochastic simulation;
- league and tournament competition resolution;
- historical replay;
- diagnostic performance evaluation.

Version 2A primarily represents a platform-engineering milestone.

The next phase, Version 2B, will focus on scientific model
improvement rather than foundational runtime construction.

---

## 2. Core Architectural Principle

The platform is scoreline-first.

The authoritative simulation sequence is:

```text
Football information
        ↓
Expected home and away goals
        ↓
Scoreline probability model
        ↓
Sampled scoreline
        ↓
Derived match outcome
        ↓
Competition progression

Match outcomes are therefore consequences of generated scorelines.

The production architecture does not normally predict a match
winner and then manufacture a compatible scoreline.

A legacy outcome-first route remains available for explicit
compatibility and historical comparison, but it is not the default
simulation mode.

The active simulation configuration uses:

MATCH_ENGINE_MODE = "scoreline_first"
3. Architectural Layers

Version 2A is organized into the following conceptual layers.

3.1 Evidence acquisition

This layer collects raw football evidence from external sources.

Examples include:

Sofascore competition data;
player profiles;
player statistics;
historical match results;
FIFA rankings;
ClubElo histories;
Opta-derived rating-prior research snapshots.

The acquisition layer does not make match predictions.

3.2 Evidence processing

Raw evidence is transformed into stable processed datasets.

Responsibilities include:

player identity reconciliation;
competition and season discovery;
position normalization;
feature construction;
historical fixture validation;
missingness auditing;
completed-match dataset construction.

The output of this layer is reproducible evidence suitable for
football-intelligence construction and model training.

3.3 Player intelligence

The player-intelligence layer converts processed player evidence
into structured player and team representations.

Important components include:

CompetitionPlayerRepository
CompetitionRosterBuilder
CompetitionTeamRepository
TeamRepresentationBuilder
PlayerRepresentationEngine
StartingXIBuilder

The principal output is a TeamRepresentation.

A team representation may include:

attack;
midfield;
defense;
goalkeeper;
attack depth;
midfield depth;
defense depth;
squad quality;
evidence score;
player-count metadata;
representation and aggregation metadata.

This layer is responsible for football intelligence, not match
simulation.

3.4 Production artifact construction

Research-domain team representations are converted into immutable
production artifacts.

The central construction flow is:

TeamRepresentation
        ↓
ProductionClubRecord
        ↓
ProductionClubRepositoryBuilder
        ↓
Versioned CSV repository

The production repository builder:

consumes existing team representations;
validates representation contracts;
serializes deterministic records;
validates persistence integrity;
writes the repository atomically.

It does not:

calculate player ratings;
build player repositories;
resolve ClubElo;
fit prediction models;
simulate matches.

This preserves a clean boundary between football-intelligence
construction and production persistence.

4. Club Production Runtime

The authoritative club-prediction runtime is:

ProductionClubRepository
        ↓
LiveMatchObservationBuilder
        ↓
ProductionGoalModel
        ↓
ProductionPredictionPipeline
4.1 ProductionClubRepository

ProductionClubRepository is a read-only runtime repository.

It:

loads a persisted club-representation artifact;
validates required values;
normalizes club names;
rejects duplicate normalized identities;
resolves clubs for live prediction.

The repository contains football-intelligence values but performs
no model inference.

4.2 ClubEloRepository

ClubEloRepository provides date-valid club-strength priors.

It:

loads cached rating histories;
resolves the rating valid on the requested prediction date;
exposes temporal validity metadata;
remains competition-agnostic.

Club-specific naming differences are supplied through explicit
name overrides.

4.3 LiveMatchObservationBuilder

LiveMatchObservationBuilder combines:

Production club representation
        +
Date-valid ClubElo rating
        ↓
LiveMatchObservation

The resulting observation contains:

resolved home and away identities;
prediction date;
attack values;
defense values;
attack-depth difference;
rating priors;
rating-prior difference;
rating effective intervals;
repository provenance.

It also validates that the observation feature mapping matches the
current production goal-model contract.

The current model feature contract is:

home_attack
away_attack
home_defense
away_defense
attack_depth_diff
rating_prior_diff
4.4 ProductionGoalModel

ProductionGoalModel is a runtime wrapper around a frozen goal-model
artifact.

It:

loads the persisted artifact;
validates artifact structure;
validates required features;
predicts lambda_home;
predicts lambda_away;
exposes model provenance.

It performs no fitting.

The expected-goal predictions are the primary model outputs.

4.5 ProductionPredictionPipeline

ProductionPredictionPipeline is the authoritative public
club-fixture prediction API.

Its primary methods are:

predict_fixture(...)
predict_fixtures(...)

For each fixture, it:

constructs a live match observation;
validates the feature contract;
predicts home and away expected goals;
calculates normalized outcome probabilities;
validates prediction consistency;
returns an immutable production prediction.

The prediction includes:

resolved fixture identities;
football-intelligence features;
rating-prior information;
lambda_home;
lambda_away;
predicted total goals;
predicted goal difference;
home-win probability;
draw probability;
away-win probability;
repository provenance;
model-artifact provenance.

The pipeline does not sample a scoreline.

This is intentional.

Deterministic prediction and stochastic simulation remain separate
responsibilities.

5. Scoreline Probability and Sampling

Two related but distinct operations exist.

5.1 Deterministic probability calculation

The scoreline probability calculator converts expected goals into
normalized outcome probabilities:

lambda_home
lambda_away
        ↓
Independent Poisson score grid
        ↓
Home-win probability
Draw probability
Away-win probability

This operation does not sample a result.

It is used for prediction, replay, and evaluation.

5.2 Stochastic scoreline realization

The simulation layer converts expected goals into one realized
scoreline.

The active sampler is configured as:

GOAL_SAMPLER = "dixon_coles_hierarchical"

The sampler may incorporate:

match-level tempo variation;
team-specific variation;
low-score Dixon–Coles correction.

It returns:

(home_goals, away_goals)

The match outcome is then inferred directly from those goals.

6. FootballModel Interface

The stable interface between football modeling and competition
simulation is:

simulate_match(
    home_team,
    away_team,
    prediction_date=...
)

This interface is more general than any specific model.

The competition framework does not need to know whether a
scoreline was generated using:

a club production repository;
ClubElo;
a national-team prior;
a frozen expected-goal artifact;
a future lineup-aware model;
a historical model.

It only requires a football model capable of returning a scoreline.

This dependency inversion allows competition logic and football
modeling to evolve independently.

7. Club Simulation Runtime

Club simulation already uses the production architecture through
the football-model adapter.

The validated club-league runtime is:

ExperimentCondition
        ↓
FootballModelAdapter
        ↓
Production club repository
        ↓
LiveMatchObservationBuilder
        ↓
Integrated club goal model
        ↓
Expected goals
        ↓
Existing Dixon–Coles scoreline sampler
        ↓
LeagueMatchSimulator
        ↓
CompetitionEngine

Study 073 validated the single-match bridge between the production
club predictor and the existing scoreline sampler.

Study 074 validated a complete calendar-aware league season,
including:

fixture generation;
prediction-date propagation;
ClubElo coverage;
live observation assembly;
production goal prediction;
scoreline generation;
fixture-population preservation;
standings resolution;
standings arithmetic.

The production club simulation bridge therefore already exists and
does not need to be rebuilt.

8. National-Team and World Cup Runtime

The World Cup simulator remains scoreline-first, but it uses a
national-team-specific expected-goal pathway.

The current national-team path is approximately:

National-team repository
        ↓
National-team expected-goal model
        ↓
Configured scoreline sampler
        ↓
Group and knockout result
        ↓
World Cup competition progression

This pathway is deliberately separate from the club production
pipeline.

The club pipeline depends on:

club representations;
club competition evidence;
ClubElo histories;
a club-trained production artifact.

The World Cup pipeline depends on:

national-team representations;
national-team rating priors;
national-team calibration;
World Cup competition rules.

The two domains share:

scoreline-first philosophy;
scoreline samplers;
competition abstractions;
match-result structures.

They do not need to share an identical feature-generation or
expected-goal model.

9. Competition Layer

The competition layer is independent of football-model internals.

Important abstractions include:

CompetitionDefinition
CompetitionBuilder
CompetitionEngine
StageDefinition
LeagueMatchSimulator
StandingsEngine
KnockoutEngine
MatchResult

Responsibilities include:

participant management;
fixture organization;
league-stage resolution;
standings calculation;
knockout progression;
bracket resolution;
stage-result persistence.

The competition layer consumes realized match scorelines.

It does not construct player representations or fit prediction
models.

10. Replay and Evaluation

Version 2A establishes a clean separation among:

Prediction generation
        ↓
Operational replay
        ↓
Performance evaluation
10.1 Operational replay

Study 083 replayed all 306 completed Bundesliga 2024–25 fixtures
through the production prediction pipeline.

The replay preserved:

event identity;
fixture information;
observed scorelines;
model features;
ClubElo values;
expected goals;
outcome probabilities;
artifact provenance;
runtime status.

The replay completed with:

306 fixtures
306 successful predictions
0 runtime failures
10.2 Performance analysis

Study 084 evaluated only the frozen replay artifact.

It did not rerun predictions.

The analysis generated:

Poisson deviance;
goal MAE and RMSE;
outcome log loss;
multiclass Brier score;
exact-score log loss;
probability calibration;
draw-rate analysis;
team-level bias;
extreme-error diagnostics.

The study was explicitly classified as an overlapping-period
diagnostic because the goal-model training cutoff overlapped the
evaluation season.

This methodological boundary prevents the results from being
misrepresented as a clean out-of-sample estimate.

11. Active and Legacy Prediction Paths

Several historical pathways remain in the repository.

They should be classified rather than removed casually.

11.1 Authoritative production club path
ProductionPredictionPipeline

Status:

AUTHORITATIVE PRODUCTION API
11.2 Club simulation adapter path
FootballModelAdapter
        ↓
Integrated club goal model
        ↓
Scoreline sampler

Status:

ACTIVE PRODUCTION SIMULATION ADAPTER
11.3 National-team scoreline-first path
National-team expected-goal model
        ↓
Scoreline sampler
        ↓
World Cup simulator

Status:

ACTIVE DOMAIN-SPECIFIC PRODUCTION PATH
11.4 IntegratedClubGoalPredictor

This is a lower-level wrapper around the frozen production goal
model.

Status:

LOWER-LEVEL PRODUCTION COMPONENT

New external runtime code should generally prefer the complete
ProductionPredictionPipeline.

11.5 Legacy ML outcome-first route

This route samples a home-win, draw, or away-win outcome before
constructing a compatible scoreline.

Status:

LEGACY COMPATIBILITY / RESEARCH COMPARISON

It remains available only when explicitly selected.

It is not the default simulation mode.

11.6 Historical inference and classifier stack

Modules under areas such as:

inference/
models/
analysis/
outputs/ml/
scripts/research/

contain valuable historical modeling and calibration work.

Status:

RESEARCH HISTORY / LEGACY MODELING STACK

These modules should remain available for reproducibility unless a
separate deprecation and archival review approves removal.

12. Confirmed Architectural Strengths

The Version 2A review confirmed the following strengths.

12.1 Separation of responsibilities

The platform separates:

evidence collection;
evidence processing;
player intelligence;
artifact construction;
runtime prediction;
scoreline simulation;
competition resolution;
replay;
evaluation.
12.2 Dependency inversion

Competition simulators depend on a generic football-model interface
rather than model internals.

12.3 Frozen production artifacts

Production inference loads versioned persisted artifacts and does
not refit models at runtime.

12.4 Explicit temporal validity

Club rating priors are resolved against fixture dates and include
effective-date intervals.

12.5 Provenance propagation

Predictions retain:

repository version;
repository scope;
model artifact name;
model version;
baseline version;
feature specification;
training cutoff.
12.6 Study-to-framework promotion

Reusable logic created during recent studies was promoted into
shared production modules rather than left inside study scripts.

12.7 Evaluation isolation

Evaluation consumes frozen prediction artifacts and does not alter
runtime prediction behavior.

13. Known Technical Debt

The following issues are non-blocking for the Version 2A freeze.

13.1 Public entry-point clarity

The project contains:

ProductionPredictionPipeline
ProductionGoalModel
IntegratedClubGoalPredictor

Their intended hierarchy should be documented clearly.

Recommended hierarchy:

ProductionPredictionPipeline
        ↓
LiveMatchObservationBuilder
        ↓
ProductionGoalModel
        ↓
Frozen artifact

IntegratedClubGoalPredictor should be documented as a lower-level
or compatibility-facing component.

13.2 Legacy pathway classification

Legacy inference, classifier, and outcome-first modules remain
intermixed with active files.

They should eventually receive explicit lifecycle labels:

active
compatibility
research-only
deprecated
archived
13.3 Competition-specific name overrides

ClubElo aliases are currently supplied through runtime mappings.

A future shared club-identity or external-provider registry may
provide a cleaner long-term location.

13.4 Repository default paths

Some lower-level runtime modules retain historical default artifact
paths.

The production pipeline factory already supports explicit paths and
should remain the preferred assembly mechanism.

13.5 Incomplete representation fields

squad_quality and evidence_score were zero for the Bundesliga
production repository.

This did not affect the current goal-model feature contract, but it
indicates an upstream football-intelligence issue that should be
investigated during Version 2B.

13.6 Model scoring-environment bias

Study 084 found systematic underprediction of Bundesliga goal
volume.

This is a scientific model issue, not an architectural blocker.

14. Version 2A Freeze Boundary

Version 2A is considered architecturally complete because it
provides:

generalized competition evidence processing;
reusable player and team intelligence;
production repository construction;
immutable production schemas;
date-aware club rating priors;
frozen goal-model artifacts;
deterministic prediction APIs;
scoreline-first stochastic simulation;
generic football-model adapters;
complete league simulation;
historical replay;
diagnostic evaluation.

The following work belongs to Version 2B:

correcting systematic goal-volume bias;
improving elite-team representation;
expected-lineup integration;
availability and injury modeling;
transfer-aware seasonal representations;
tactical context;
dynamic form integration into production;
clean cross-league and forward-period evaluation;
improved uncertainty and calibration;
further national-team production-model development.
15. Final Architectural Verdict
VERSION 2A ARCHITECTURE: COMPLETE
PRODUCTION FREEZE STATUS: APPROVED

The project now possesses a coherent end-to-end architecture:

Evidence
        ↓
Player intelligence
        ↓
Team representation
        ↓
Production artifacts
        ↓
Expected-goal prediction
        ↓
Scoreline-first simulation
        ↓
Competition resolution
        ↓
Replay
        ↓
Evaluation

Future development should preserve these boundaries.

Version 2B should improve the football model within this
architecture rather than redesigning the architecture without
strong evidence that a boundary has failed.