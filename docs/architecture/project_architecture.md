# World Cup 2026 Football Simulation Framework — Project Architecture

## 1. Purpose and Architectural Principles

This document describes the current architecture of the World Cup 2026 Football Simulation Framework at the time of the Version 1 release-preparation phase.

Its purpose is to distinguish clearly between:

* current production runtime paths;
* validated but not yet authoritative architecture;
* research and experimental infrastructure;
* compatibility layers retained from earlier versions;
* generated data and external-source boundaries.

The project began as a World Cup simulation system but has evolved into a broader computational football research platform. Its architecture therefore contains both operational simulation infrastructure and a substantial research environment used to evaluate candidate representations, models, and competition abstractions.

This document describes **what the project currently does**, not every architecture that has been investigated.

### 1.1 Core Architectural Principles

The project follows several principles.

**Separation of concerns**

Data acquisition, player intelligence, team representation, football modeling, scoreline realization, competition logic, and research infrastructure should remain conceptually separate.

**Stable interfaces between layers**

A downstream component should depend on a defined football-domain interface rather than on the implementation details of an upstream data source or research method.

Examples include:

```text
Player evidence
        ↓
canonical player representation

Team representation
        ↓
team repository

Football model
        ↓
lambda_home / lambda_away

Match realization
        ↓
scoreline
```

**Research before production promotion**

The presence of a research implementation does not imply that it belongs to the production runtime.

Alternative aggregation strategies, contextual representations, formation geometry, structural responsibility models, and other experimental mechanisms remain research-only until empirical validation justifies promotion.

**Preserve validated behavior**

Architectural generalization should not replace a validated production path merely for conceptual elegance.

Migration occurs only when the replacement can reproduce or improve required behavior without breaking established interfaces.

**Explicit provenance**

Football data is treated as evidence with source, temporal, identity, and competition context.

Historical evidence should not be collapsed prematurely when its competition-season provenance still matters.

**Reproducible methodology rather than repository completeness**

The repository is not intended to contain every raw or generated dataset.

Instead, it versions the source code, model-defining configuration, reviewed mappings, methodology, tests, and small reproducibility artifacts required to reconstruct the system where external data remains available.

---

## 2. System-Level Architecture

At the highest level, the project can be represented as six sequential runtime layers surrounded by a separate research and validation plane.

```text
┌─────────────────────────────────────────────────────────────┐
│                  RESEARCH AND VALIDATION                    │
│                                                             │
│  studies · benchmarks · replay · calibration · ablation    │
│  tests · diagnostics · promotion gates · prototypes        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ evaluates
                         ▼

1. DATA ACQUISITION
   External football sources
   Competition and season discovery
   Player and team ingestion
                │
                ▼
2. PLAYER INTELLIGENCE
   Canonical player identity
   Historical competition-season evidence
   Player features
   Player attributes
   Role ratings
                │
                ▼
3. TEAM REPRESENTATION
   Squad and lineup construction
   Player aggregation
   Team-strength representation
   Contextual representation where validated
                │
                ▼
4. FOOTBALL MODEL
   Match observation construction
   Team-strength priors
   Expected-goal estimation
                │
                ▼
          lambda_home
          lambda_away
                │
                ▼
5. MATCH REALIZATION
   Goal sampling
   Scoreline generation
                │
                ▼
6. COMPETITION AND MONTE CARLO
   Group stages
   Standings
   Advancement
   Knockout brackets
   Tournament realization
   Repeated simulation
   Observers and probability summaries
```

### 2.1 The Lambda Boundary

The most important convergence boundary in the current architecture is:

```text
lambda_home
lambda_away
```

The football model is responsible for estimating expected goals.

The match-realization layer is responsible for converting those expected goals into a stochastic scoreline.

This separation allows the football model to evolve without requiring the scoreline-sampling layer to be redesigned.

The current repository contains more than one upstream route to expected goals:

```text
National-team / World Cup route
        ↓
calibrated lambda model
        ↓
lambda_home / lambda_away
```

and:

```text
Modern club production route
        ↓
artifact-backed ProductionGoalModel
        ↓
lambda_home / lambda_away
```

These paths are not yet fully unified upstream, but they converge on the same conceptual expected-goals boundary.

### 2.2 Production and Research Are Not the Same Axis

The `research/` directory is not simply an earlier version of the production system.

Research infrastructure evaluates hypotheses about:

* player evidence;
* team representation;
* aggregation mathematics;
* expected-lineup selection;
* rating priors;
* contextual realization;
* goal models;
* calibration;
* competition structure;
* formation geometry;
* structural responsibility;
* football observability.

A successful research result may eventually be promoted into a production-facing interface, but most studies remain isolated from runtime behavior until that promotion occurs.

This distinction is essential when interpreting the repository.

---

## 3. Data Acquisition and Provenance Boundary

The project is designed to be **source-reconstructible rather than self-contained**.

Large provider-derived datasets, processed research datasets, generated model artifacts, and simulation outputs are generally excluded from version control.

The repository instead retains acquisition logic, reviewed mappings, manifests, configuration, validation rules, and other small artifacts required to reconstruct those datasets where the external source remains available.

### 3.1 Acquisition Architecture

The modern player-data acquisition path is approximately:

```text
Competition and season configuration
        ↓
competition_manifest.csv
        ↓
Player competition-season discovery
        ↓
sofascore_players.csv
        ├────────────────────┐
        │                    │
        ▼                    ▼
Player profile ingestion     Player statistics ingestion
        │                    │
        ▼                    ▼
sofascore_player_profiles.csv
                             sofascore_player_stats.csv
```

The generated player datasets are local artifacts and are not intended to be committed to Git.

The ingestion scripts support explicit competition-season scope, configurable input/output paths, and resumable acquisition.

Historical player statistics remain associated with their source competition and season until later evidence aggregation.

### 3.2 Competition Manifest

The competition manifest is a version-controlled acquisition artifact.

It combines domestic and international competition-season registries and encodes information including:

* competition identity;
* competition type;
* season identity;
* recency weighting;
* competition importance;
* ingestion priority;
* acquisition flags.

For the current production domestic scope, the manifest builder enforces the Big Five league contract unless an explicit research mode is used.

The manifest therefore belongs to the modeling and reproducibility surface rather than being treated as disposable scraped data.

### 3.3 Identity and Evidence Are Separate

The data architecture distinguishes between:

```text
Who is the player?
```

and:

```text
What historical football evidence belongs to that player?
```

Current player profiles contribute to canonical identity.

Historical competition-season statistics contribute to football evidence.

Reviewed player-ID aliases are applied when resolving source evidence to canonical player identity.

The canonical evidence grain is:

```text
competition × season × canonical player
```

This grain is preserved until historical evidence weighting has been performed.

### 3.4 External Source Boundary

The acquisition layer currently depends substantially on external football data providers, including Sofascore and rating-prior sources.

This creates an explicit external boundary:

```text
Version-controlled methodology
        ↓
External provider
        ↓
Local raw data
        ↓
Validated project transformations
        ↓
Local processed artifacts
```

Exact future reconstruction cannot be guaranteed because external services may change:

* endpoints;
* schemas;
* historical coverage;
* availability;
* rate limits;
* identifiers;
* source values.

The architecture therefore aims to make acquisition and transformation logic reproducible while acknowledging that external football data itself is not under project control.

### 3.5 Version-Controlled Data Surface

Small artifacts are retained when they define project behavior or are important for reproducibility.

Examples include:

```text
competition_manifest.csv
competition_feature_manifest.csv
stat_manifest.csv

feature_attribute_manifest.csv
role_attribute_manifest.csv
role_feature_manifest.csv

formation_manifest.csv
formation_geometry.csv

sofascore_player_id_aliases.csv
```

Large raw provider datasets and generated outputs are generally excluded.

This distinction allows the repository to preserve methodological reproducibility without turning Git into a storage system for scraped or derived football data.

## 4. Player Intelligence Architecture

Player Intelligence is responsible for transforming provider-derived player evidence into canonical football representations that can later contribute to team strength.

The production architecture separates three concepts that earlier versions of the project sometimes treated more closely together:

```text
Player identity
        │
        ├──────────────┐
        │              │
        ▼              ▼
Current identity   Historical evidence
information       competition × season × player
        │              │
        ▼              ▼
Canonical player  Canonical evidence resolution
registry                │
                        ▼
                 Historical evidence weighting
                        │
                        ▼
                 Player football features
                        │
                        ▼
                 Player attributes
                        │
                        ▼
                   Role ratings
```

### 4.1 Canonical Identity

Current player identity is represented separately from historical statistical evidence.

The canonical player registry is built from current player-profile information and provides the identity surface used by downstream Player Intelligence components.

Historical source identifiers are not assumed to be permanently reliable.

Reviewed aliases may therefore map source player identifiers onto canonical player identifiers.

This separation prevents current identity metadata from being incorrectly interpreted as historical competition membership.

### 4.2 Canonical Historical Evidence

Historical player evidence is resolved independently from the current-profile registry.

The authoritative historical evidence key is:

```text
competition_id
× season_id
× canonical_player_id
```

Source rows are first mapped onto canonical identities.

Where multiple source identifiers collapse onto the same canonical task key, the resolver validates that the underlying football evidence is equivalent before deduplicating the collision.

Historical evidence is therefore canonicalized without silently combining conflicting observations.

### 4.3 Evidence Weighting

Competition-season evidence remains disaggregated until weighting has been applied.

Conceptually:

```text
Canonical competition-season evidence
        ↓
competition importance
        +
recency
        +
evidence/sample considerations
        ↓
weighted historical player representation
```

This prevents premature aggregation from destroying the historical provenance needed to distinguish recent, old, high-value, and lower-value evidence.

### 4.4 Features, Attributes, and Roles

The downstream Player Intelligence transformation is:

```text
weighted player evidence
        ↓
canonical player features
        ↓
football attributes
        ↓
role ratings
```

Feature definitions, attribute mappings, and role mappings are manifest-driven where appropriate.

The resulting player representation is intended to describe football ability rather than reproduce the source provider's raw statistical schema.

### 4.5 Authoritative Production Pipeline

The authoritative Player Intelligence orchestration entry point is:

```bash
python -m scripts.run_player_intelligence_pipeline
```

The production sequence is:

```text
resolve_player_evidence
        ↓
build_weighted_player_features
        ↓
score_player_attributes
        ↓
build_player_ratings_v4
```

Study 101F froze this ordering as the current production Player Intelligence architecture.

Earlier Player Intelligence builders remain in the repository for compatibility, research history, or comparison but should not be interpreted as the authoritative production path.

---

## 5. Team Representation Boundary

Player Intelligence answers:

```text
What do we know about individual players?
```

Team representation answers a different question:

```text
What does the available player population imply about the strength
and structure of a football team?
```

This boundary is intentionally explicit.

### 5.1 Representation Responsibilities

Team representation may involve:

* squad construction;
* expected-lineup selection;
* positional and role suitability;
* aggregation of player ability;
* depth measurement;
* distribution-sensitive aggregation;
* contextual realization;
* formation-aware interpretation.

Not all researched representation mechanisms are currently part of production.

The presence of an implementation under `research/player_intelligence/` or `research/studies/` does not by itself indicate production promotion.

### 5.2 Canonical Repository Interface

For the current national-team simulation path, downstream simulation consumes a canonical team repository.

Its public schema includes:

```text
attack
midfield
defense
gk

poisson_attack
poisson_defense

rating_prior
```

Compatibility aliases remain temporarily available for older runtime components.

The repository interface is intended to isolate the match engine from the specific method used to construct team strength.

Conceptually:

```text
Legacy team strength
        │
Dimension-specific aggregation
        │
Expected-lineup representation
        │
Future validated representation
        │
        ▼
Canonical Team Repository
        │
        ▼
Downstream football model
```

The backing representation can therefore evolve without requiring tournament logic to understand Player Intelligence internals.

### 5.3 Current World Cup Repository State

The current World Cup runtime remains configured with:

```text
TEAM_REPOSITORY_SOURCE = "legacy"
```

This is a deliberate production-state distinction.

More sophisticated representations have been researched, but the existence of those representations does not automatically replace the established World Cup baseline.

Promotion requires explicit validation.

### 5.4 Modern Club Representation

The newer club production architecture uses a more focused runtime representation.

The current required club fields are:

```text
attack
defense
attack_depth
```

The repository may also carry additional information such as:

```text
midfield
goalkeeper
midfield_depth
defense_depth
squad_quality
evidence_score
```

These additional fields do not automatically enter the production goal model.

They remain available for provenance, diagnostics, and future validated model development.

### 5.5 Research Frontier

Recent team-representation research has investigated:

* expected-XI selection;
* aggregation mathematics;
* transformation and scale compatibility;
* contextual realization;
* role suitability;
* formation geometry;
* positional responsibility;
* structural defensive relationships;
* cross-formation behavior.

These studies extend the representational vocabulary of the project but remain subject to research-before-promotion rules.

---

## 6. Football-Model Architecture

The football-model layer converts team-level football information into expected goals.

Its output contract is:

```text
lambda_home
lambda_away
```

This expected-goals boundary separates football prediction from stochastic scoreline realization.

### 6.1 Current World Cup Lambda Path

The configured World Cup path uses the canonical team repository through the match-engine adapter.

The effective runtime is:

```text
Canonical Team Repository
        ↓
repository_entry_to_poisson_features
        ↓
poisson_attack
poisson_defense
rating_prior
        ↓
calibrated_expected_goals
        ↓
lambda_home
lambda_away
```

The calibrated lambda model is a fitted log-link Poisson model.

Its current conceptual inputs are:

```text
home attacking strength
away defensive strength

away attacking strength
home defensive strength

external rating-prior difference
```

The coefficient artifact retains historical FIFA-oriented naming because the model was originally fitted using FIFA points for national teams.

The runtime interface now exposes this signal generically as:

```text
rating_prior
```

### 6.2 Legacy Heuristic Lambda Model

A hand-tuned heuristic expected-goals implementation remains available for compatibility and comparison.

It is not the currently configured production lambda model.

The production configuration selects:

```text
LAMBDA_MODEL = "calibrated"
```

### 6.3 Generic Goal-Model Research Interface

The repository also contains a generic `GoalModel` abstraction and `PoissonGoalModel` implementation.

These support model fitting and evaluation over configurable feature sets.

They should be understood primarily as modeling and research infrastructure rather than as the authoritative World Cup runtime interface.

### 6.4 Artifact-Backed Production Goal Model

The newer club prediction architecture uses a frozen artifact-backed production interface.

Conceptually:

```text
Validated football representation
        +
prediction-date external rating prior
        ↓
LiveMatchObservation
        ↓
feature mapping
        ↓
ProductionGoalModel
        ↓
lambda_home
lambda_away
```

The model artifact declares the features required for inference.

The runtime validates that the supplied feature mapping satisfies that contract before prediction.

This architecture separates:

```text
model fitting
```

from:

```text
production inference
```

and preserves artifact provenance such as model version, baseline version, feature specification, and training cutoff.

### 6.5 Current Club Production Feature Contract

The currently promoted club model consumes:

```text
home_attack
away_attack

home_defense
away_defense

attack_depth_diff

rating_prior_diff
```

The rating prior is resolved for the prediction date using ClubElo.

The observation layer validates temporal applicability before the feature mapping reaches the production model.

### 6.6 Production Prediction Pipeline

The modern deterministic club prediction path is:

```text
ProductionClubRepository
        +
ClubEloRepository
        ↓
LiveMatchObservationBuilder
        ↓
LiveMatchObservation
        ↓
ProductionGoalModel
        ↓
lambda_home / lambda_away
        ↓
outcome probability calculation
        ↓
ProductionFixturePrediction
```

The production prediction pipeline deliberately does not:

* build Player Intelligence repositories;
* download rating histories;
* fit goal models;
* sample stochastic scorelines;
* evaluate model accuracy.

Those responsibilities remain outside the deterministic prediction boundary.

### 6.7 Current Dual Production State

Version 1 contains two upstream production-generation paths.

```text
WORLD CUP / NATIONAL TEAM

Canonical Team Repository
        ↓
fixed calibrated lambda interface
        ↓
lambda_home / lambda_away
```

and:

```text
MODERN CLUB PREDICTION

Production Club Repository
        +
prediction-date ClubElo
        ↓
artifact-backed feature contract
        ↓
lambda_home / lambda_away
```

These paths are not yet unified upstream.

They converge at the expected-goals boundary.

This is an intentional description of the current architecture rather than a claim that the two systems have already been consolidated.

### 6.8 Architectural Convergence Point

The long-term football-model architecture can evolve above the lambda boundary without requiring scoreline realization to change.

A future validated national-team model may therefore use richer Player Intelligence features while still returning:

```text
lambda_home
lambda_away
```

to the same downstream match-realization interface.

Such migration is not a Version 1 release requirement.

## 7. Match-Realization Architecture

The match-realization layer is responsible for converting expected goals into a stochastic football scoreline.

It begins only after the football model has produced:

```text
lambda_home
lambda_away
```

This creates a deliberate architectural boundary:

```text
Football representation
        ↓
Expected-goal model
        ↓
lambda_home / lambda_away
────────────────────────────────
     MATCH-REALIZATION BOUNDARY
────────────────────────────────
        ↓
Goal sampler
        ↓
Scoreline
```

### 7.1 Current Production Mode

The current simulation configuration selects:

```text
MATCH_ENGINE_MODE = "scoreline_first"
GOAL_SAMPLER = "dixon_coles_hierarchical"
LAMBDA_SCALE = 0.75
```

The configured scoreline-first route therefore:

1. obtains expected goals;
2. applies the configured lambda scale;
3. passes the scaled expected goals to the production goal sampler;
4. returns the realized integer scoreline.

### 7.2 Scoreline-First Principle

The current production architecture treats the scoreline as the primary stochastic football realization.

Match outcome is therefore derived from the generated scoreline rather than sampled first and followed by a compatible scoreline.

Conceptually:

```text
Expected goals
      ↓
Scoreline
      ↓
win / draw / loss
```

rather than:

```text
win / draw / loss
      ↓
forced compatible scoreline
```

This preserves a coherent relationship between expected goals, scoreline distribution, and match outcome.

### 7.3 Dixon-Coles Hierarchical Realization

The currently configured production sampler is the Dixon-Coles hierarchical sampler.

Its configuration includes:

```text
tempo_cv
team_cv
rho
```

The sampler therefore operates downstream of expected-goal estimation and can be evaluated independently from the football model that produced those expected goals.

This separation has been important throughout goal-sampler research because alternative realization mechanisms can be benchmarked without changing the upstream team representation.

### 7.4 External Lambda Entry Point

The match-engine adapter exposes a direct scoreline interface for externally supplied expected goals:

```text
simulate_scoreline_from_lambdas(
    lambda_home,
    lambda_away,
)
```

This is an important integration seam.

It allows an artifact-backed production predictor, or any future validated football model, to replace the source of expected goals while preserving the existing production scoreline-realization layer.

### 7.5 Legacy ML-Guided Route

A legacy ML-guided match route remains available explicitly through:

```text
mode = "ml"
```

This path samples an outcome from predicted outcome probabilities and then searches for a compatible Poisson scoreline.

It is retained for compatibility and historical comparison.

It is not the currently configured production match-engine mode.

---

## 8. Competition Architecture

Competition logic is conceptually separate from football-model logic.

A competition engine should determine:

* which fixtures exist;
* how results affect standings;
* which participants advance;
* how knockout stages are constructed;
* how a champion is determined.

It should not need to understand how team strength was calculated or how a match scoreline was generated.

### 8.1 Generic Competition Framework

The repository contains generic abstractions for:

```text
Competition
Stage
StageResult
MatchResult
Standings
Advancement
Bracket
Tie
KnockoutEngine
StageResolver
CompetitionEngine
```

Supporting infrastructure also exists for reusable competition definitions and fixture generation.

The intended architectural relationship is:

```text
Competition definition
        ↓
Stages and fixtures
        ↓
Match results
        ↓
Stage resolution
        ↓
Standings / advancement
        ↓
Competition result
```

### 8.2 CompetitionEngine Version 1

`CompetitionEngine` provides a minimal generic stage-resolution pipeline.

For each configured stage it:

1. resolves the stage through `StageResolver`;
2. stores the resulting stage output;
3. applies the stage advancement rule where one exists;
4. records advancement information;
5. infers a simple champion and runner-up from the final resolved stage where possible.

Version 1 intentionally does not dynamically generate all future competition stages.

The generic framework should therefore be treated as validated competition infrastructure, but not as a claim that every production competition has already migrated onto it.

### 8.3 Competition Catalog and Fixture Generation

Reusable competition definitions live separately from the competition engine.

The repository currently includes competition definitions for:

```text
Premier League
2026 FIFA World Cup
```

Generic round-robin fixture generation also exists outside the competition-resolution layer.

This separation allows competition identity and scheduling rules to evolve independently from match prediction.

### 8.4 Current World Cup Production Orchestration

The authoritative World Cup tournament runtime remains World Cup-specific.

Its current sequence is:

```text
World Cup group-stage simulation
        ↓
Group standings
        ↓
Qualifier extraction
        ↓
Official Round-of-32 mapping
        ↓
Round of 32
        ↓
Round of 16
        ↓
Quarterfinals
        ↓
Semifinals
        ├───────────────┐
        ▼               ▼
Third-place match      Final
        │               │
        └───────┬───────┘
                ▼
         TournamentResult
```

The current `wc2026_tournament_simulator.py` directly orchestrates these World Cup-specific components.

It does not use `CompetitionEngine` as its authoritative production runtime.

### 8.5 Current Dual Competition State

Version 1 therefore contains:

```text
GENERIC COMPETITION ARCHITECTURE

competition_catalog/
fixture_generation/
simulation/competition/
```

alongside:

```text
CURRENT WORLD CUP PRODUCTION ORCHESTRATION

scripts/wc2026_group_stage.py
scripts/wc2026_knockout_mapping.py
scripts/wc2026_knockout_stage.py
scripts/wc2026_tournament_simulator.py
```

This coexistence is intentional for the Version 1 release snapshot.

Migrating the World Cup onto the generic engine is an architectural convergence opportunity, not a release requirement.

---

## 9. Monte Carlo and Observation Layer

A single tournament realization is not treated as a probabilistic forecast.

The Monte Carlo layer repeatedly executes the complete tournament and aggregates the resulting event frequencies.

### 9.1 Monte Carlo Runtime

The current World Cup Monte Carlo sequence is:

```text
Canonical Team Repository
        ↓
simulate_tournament()
        ↓
TournamentResult
        ↓
repeat N times
        ↓
event counters
        +
simulation observers
        ↓
probability tables
        +
simulation diagnostics
```

The driver supports deterministic random seeding for repeatable simulation runs.

### 9.2 Tournament Probability Outputs

The current driver aggregates events including:

```text
champion
runner-up
semifinal
quarterfinal
round of 16
```

Event counts are converted into empirical Monte Carlo probabilities:

```text
event probability
=
event count / number of simulated tournaments
```

These probabilities describe the behavior of the configured model under repeated simulation.

They are not guarantees about real tournament outcomes.

### 9.3 Observer Architecture

Tournament observation is separated from tournament execution.

The Monte Carlo driver currently uses an observer manager with observers including:

```text
StatisticsObserver
ExtremeEventsObserver
```

This allows simulation behavior to be measured without embedding diagnostic logic directly inside tournament orchestration.

The observer layer can therefore record quantities such as:

* total goals;
* goals per match;
* extra-time frequency;
* penalty-shootout frequency;
* extreme simulated events.

This separation supports both production diagnostics and research analysis.

### 9.4 Generated Outputs

Monte Carlo results are written as generated artifacts under the output tree.

Typical outputs include:

```text
champion_probabilities.csv
runner_up_probabilities.csv
semifinal_probabilities.csv
quarterfinal_probabilities.csv
round_of_16_probabilities.csv
simulation_statistics.csv
extreme_event_leaderboards.csv
```

These outputs are not part of the version-controlled source architecture.

They are reproducible products of a particular model configuration, repository state, simulation count, and random seed.

---

## 10. Research and Validation Plane

Research and validation surround the runtime architecture rather than forming one sequential production stage.

Conceptually:

```text
                  RESEARCH / VALIDATION
        ┌─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
Data → Player Intelligence → Team → Football Model → Match → Competition
        ▲                                     ▲
        │                                     │
        └─────────────────────────────────────┘
```

Research may interrogate any boundary in the system.

### 10.1 Research Responsibilities

The research environment supports activities including:

* controlled experiments;
* benchmarking;
* historical replay;
* feature ablation;
* representation comparison;
* calibration analysis;
* model sensitivity analysis;
* competition-format research;
* synthetic scenario evaluation;
* football-observability studies;
* production-candidate validation.

Research implementations primarily live under:

```text
research/
```

with additional operational research scripts under:

```text
scripts/research/
```

### 10.2 Automated Testing

Conventional software invariants are tested under:

```text
tests/
```

The test suite covers areas including:

* aggregation behavior;
* Player Intelligence integration;
* historical compatibility;
* lineup-assignment preservation;
* player contribution;
* role suitability;
* formation geometry;
* structural responsibility;
* team-representation integration;
* research-study utilities;
* operational repository builders.

Tests answer questions such as:

```text
Does the implementation obey its defined contract?
```

They do not replace empirical football-model validation.

### 10.3 Validation and Benchmarking

Football-model questions frequently require empirical validation rather than only unit tests.

Examples include:

```text
Does this representation improve prediction?
Does this model remain calibrated?
Does the behavior generalize across competitions?
Does the change alter tournament behavior unexpectedly?
Does the new signal provide information beyond the existing baseline?
```

These questions are evaluated through studies, replay, benchmarking, calibration, and sensitivity analysis.

### 10.4 Promotion Principle

The project uses the following conceptual promotion path:

```text
Hypothesis
    ↓
Research implementation
    ↓
Controlled evaluation
    ↓
Validation
    ↓
Production candidate
    ↓
Promotion decision
    ↓
Production interface
```

Not every successful experiment requires production promotion.

A new component should enter the runtime only when it provides sufficient value to justify the additional architectural complexity.

### 10.5 Freeze and Decision Artifacts

Where research produces a lasting architectural decision, the repository may retain:

* study protocols;
* decision documents;
* freeze notes;
* validation scripts;
* small metadata artifacts;
* tests enforcing the promoted behavior.

Study 101F is an example of this pattern for the current Player Intelligence production pipeline.

This preserves the reasoning behind important architectural transitions without requiring generated research outputs to remain in version control.

## 11. Production, Research, Compatibility, and Legacy Classification

The repository contains multiple generations of football infrastructure.

A component's presence in the repository does not by itself identify its architectural status.

For the Version 1 release snapshot, components should be interpreted through four broad classifications.

### 11.1 Production

Production components are part of an authoritative or explicitly supported runtime path.

Current examples include:

```text
Player Intelligence
    scripts/run_player_intelligence_pipeline.py
    scripts/resolve_player_evidence.py
    scripts/build_weighted_player_features.py
    scripts/score_player_attributes.py
    scripts/build_player_ratings_v4.py

National-team simulation
    scripts/team_strength_loader.py
    simulation/match_engine_adapter.py
    simulation/lambda_models.py
    simulation/goal_samplers.py

Club prediction
    simulation/production_goal_model.py
    simulation/live_match_observation_builder.py
    research/production/production_prediction_pipeline.py

World Cup simulation
    scripts/wc2026_group_stage.py
    scripts/wc2026_knockout_mapping.py
    scripts/wc2026_knockout_stage.py
    scripts/wc2026_tournament_simulator.py
    scripts/monte_carlo_driver.py
```

Production status does not imply that every production subsystem has already been consolidated onto one common architecture.

### 11.2 Validated Architecture Not Yet Authoritative for All Runtimes

Some components represent deliberate architectural development and have substantial validation but have not replaced every existing runtime.

Important examples include:

```text
competition_catalog/
fixture_generation/
simulation/competition/
```

These provide reusable competition abstractions and generic competition infrastructure.

They should not be described as the authoritative World Cup tournament runtime until that migration has actually occurred.

### 11.3 Research

Research components investigate candidate football mechanisms, representations, models, and architectural changes.

Examples include research into:

```text
expected-lineup selection
aggregation mathematics
representation calibration
context realization
role suitability
formation geometry
structural responsibility
cross-formation behavior
football evidence and observability
```

Research status means:

```text
implemented and investigated
```

not necessarily:

```text
selected for production
```

### 11.4 Compatibility and Legacy

Earlier implementations remain where they provide:

* backward compatibility;
* reproducibility of earlier studies;
* benchmark baselines;
* comparison against promoted architecture;
* transitional support for older interfaces.

Examples include earlier player-rating builders, older aggregation strategies, heuristic expected-goal logic, and the explicit ML-guided match-engine mode.

Compatibility code should not be confused with the preferred architecture merely because it remains executable.

---

## 12. Current Architectural Convergence Points

The Version 1 architecture contains several areas where independently developed subsystems now approach a common boundary.

These are architectural opportunities rather than release blockers.

### 12.1 Football-Model Convergence

The current national-team and club prediction systems use different upstream expected-goal architectures.

```text
NATIONAL TEAM

Canonical Team Repository
        ↓
fixed Poisson/rating-prior feature interface
        ↓
calibrated lambda model
        ↓
lambda_home / lambda_away
```

```text
CLUB

Production Club Repository
        +
prediction-date ClubElo
        ↓
LiveMatchObservation
        ↓
artifact-backed feature contract
        ↓
ProductionGoalModel
        ↓
lambda_home / lambda_away
```

Both systems already converge on the same conceptual expected-goals output.

A future architecture may generalize the artifact-backed observation/model pattern across club and national-team football.

Version 1 does not require this migration.

### 12.2 Player Intelligence to Football-Model Convergence

Player Intelligence can represent substantially more football information than the current World Cup lambda model consumes.

The national-team runtime currently reduces team representation primarily to:

```text
poisson_attack
poisson_defense
rating_prior
```

Meanwhile research and newer production representations can describe:

```text
attack
midfield
defense
goalkeeper

depth

role suitability
lineup structure
contextual realization
formation relationships
structural responsibility
```

The existence of richer information does not establish that the goal model should consume all of it.

The post-Version-1 research question is therefore:

```text
Which player- and team-derived signals provide stable incremental
predictive information beyond the existing football-model baseline?
```

Only validated signals should cross the production football-model boundary.

### 12.3 Competition-Runtime Convergence

The project contains both:

```text
generic competition infrastructure
```

and:

```text
World Cup-specific production orchestration
```

The generic framework provides cleaner reusable abstractions, while the World Cup-specific path preserves extensively validated tournament behavior.

A future migration should occur only when equivalence can be demonstrated for:

* group-stage behavior;
* standings;
* advancement;
* third-place qualification;
* official knockout mapping;
* extra-time and penalties;
* third-place playoff;
* final placement;
* Monte Carlo outputs.

Architectural cleanliness alone is not sufficient justification for migration.

### 12.4 Prediction-to-Realization Convergence

The modern deterministic production prediction pipeline stops at expected goals and outcome probabilities.

The match-engine adapter separately exposes stochastic realization from externally supplied lambdas.

These interfaces are naturally compatible:

```text
Production prediction
        ↓
lambda_home / lambda_away
        ↓
simulate_scoreline_from_lambdas
        ↓
production goal sampler
        ↓
scoreline
```

This provides a clean route for future integration without coupling production feature construction to stochastic match realization.

---

## 13. Known Architectural Debt and Deferred Work

Version 1 intentionally does not attempt to remove all historical or transitional architecture.

The following items are known areas for later consolidation.

### 13.1 Parallel National-Team and Club Prediction Paths

National-team and club prediction currently use different upstream goal-model interfaces.

This is acceptable for Version 1 but increases conceptual duplication.

A later version may define a common production observation and artifact interface where empirical evidence supports doing so.

### 13.2 Parallel Competition Runtimes

The generic competition framework and the current World Cup-specific orchestration coexist.

The World Cup-specific path remains authoritative for the Version 1 tournament simulator.

Migration should be treated as a dedicated equivalence project rather than routine refactoring.

### 13.3 Historical Repository Strategies

The team-repository loader retains multiple earlier representation strategies and validation repositories.

These are useful for reproducibility and comparison but make the configuration surface broader than the current production architecture requires.

Later cleanup may distinguish more explicitly between:

```text
production repositories
research repositories
validation fixtures
historical baselines
```

### 13.4 Compatibility Aliases

Some interfaces retain historical field names such as:

```text
fifa_points
att_composite
mid_composite
def_composite
gk_composite
poisson_attack_adj
poisson_defense_adj
```

while newer interfaces expose more generic names such as:

```text
rating_prior
attack
midfield
defense
gk
poisson_attack
poisson_defense
```

These aliases reduce migration risk but should eventually be retired when downstream compatibility no longer requires them.

### 13.5 Research Documentation Accumulation

The repository intentionally preserves substantial research history.

As the project matures, documentation should increasingly distinguish:

```text
current architecture
historical architecture
research methodology
study-specific decisions
```

without deleting the scientific history that explains why current production choices exist.

### 13.6 External Data Dependency

The repository does not control the continued availability or schema stability of external football data.

Acquisition therefore remains an operational dependency and a reproducibility limitation.

Provider changes should be handled at the acquisition boundary rather than propagated into downstream football-domain interfaces wherever possible.

---

## 14. Version 1 Freeze and Post-Release Architectural Frontier

### 14.1 Version 1 Release Principle

The Version 1 release is intended to publish a coherent, validated snapshot of the project rather than complete every architectural migration currently visible in the research roadmap.

The release should prioritize:

```text
working end-to-end behavior
clear supported entry points
validated production paths
reproducible methodology
honest documentation
```

over:

```text
architectural perfection
removal of all legacy code
promotion of every successful research idea
unification of every runtime
```

### 14.2 Version 1 Production Freeze

For release preparation, the following principles apply:

* Study 101F remains the authoritative Player Intelligence production pipeline.
* The current World Cup team-repository configuration remains unchanged unless a release-blocking defect is discovered.
* The calibrated national-team lambda model remains the World Cup expected-goals baseline.
* The Dixon-Coles hierarchical scoreline-first realization remains the configured production match sampler.
* The World Cup-specific tournament simulator remains the authoritative World Cup competition runtime.
* The generic competition framework remains available and documented without forcing a pre-release migration.
* The modern artifact-backed club prediction pipeline remains a distinct supported production architecture.
* Studies 102–106 and other newer research are not promoted merely to increase Version 1 feature count.

### 14.3 Release Blockers Versus Post-Release Work

A change is a Version 1 release blocker when it affects:

* correctness of a supported runtime;
* ability to execute a documented entry point;
* reproducibility of a claimed production pipeline;
* tournament-rule correctness;
* serious data/provenance integrity;
* dependency installation;
* public-release safety.

A change is generally post-release architectural work when it primarily concerns:

* cleaner abstraction;
* removal of duplication;
* promotion of unproven richer features;
* replacement of a validated runtime with a more generic one;
* additional research scope;
* speculative model complexity.

This distinction is intended to prevent continuing architectural research from indefinitely delaying publication.

### 14.4 Immediate Release Path

Following this architecture snapshot, Version 1 preparation should proceed through:

```text
Architecture freeze
        ↓
Release-blocker audit
        ↓
Supported-entry-point validation
        ↓
End-to-end simulator validation
        ↓
Minimal publication cleanup
        ↓
Version 1 release
```

### 14.5 Post-Version-1 Frontier

After Version 1 is published, the highest-value architectural questions include:

**Unified football-model interface**

Can national-team prediction adopt the artifact-backed observation/model architecture already used by the modern club pipeline?

**Incremental Player Intelligence value**

Which richer Player Intelligence signals provide robust predictive information beyond attack, defense, depth, and external rating priors?

**Competition-engine migration**

Can the generic competition framework reproduce the complete World Cup runtime exactly enough to become authoritative?

**Structural football representation**

Do formation geometry, positional relationships, and structural-responsibility signals improve football prediction or representation sufficiently to justify production complexity?

**Football evidence**

Which football concepts are genuinely observable in the available data, and which should remain conceptual rather than computational?

These questions form the post-release research roadmap.

They are deliberately not prerequisites for publishing Version 1.

## Current Architecture

```text
Data Collection
       |
       v
Historical Dataset
       |
       +-------------------+
       |                   |
       v                   v
Expected Goals Model   Outcome Probability Model
       |                   |
       +---------+---------+
                 |
                 v
            Match Engine
                 |
                 v
          Competition Engine
                 |
                 v
          Monte Carlo Engine
                 |
                 v
        Benchmarks and Reports

Major Packages
simulation/

Production tournament and match simulation components.

Responsibilities:

goal sampling
expected-goals integration
match sampling
group-stage simulation
knockout-stage simulation
tournament simulation
central simulation configuration

Status:

Production-critical.

inference/

Production machine-learning inference layer.

Responsibilities:

feature construction for outcome prediction
feature-vector ordering
production model loading
match probability prediction
probability engine adapter

Status:

Production-critical.

shared/

Reusable project infrastructure.

Responsibilities:

dataset preparation
feature sets
label encoding
model configuration
metrics
evaluation
feature importance
competition registry
shared outcome evaluation

Status:

Core reusable infrastructure.

scripts/

Execution layer for project workflows.

Contains:

audits
benchmarks
research experiments
data collection utilities
historical validation scripts
World Cup construction scripts

Status:

Mixed. Needs continued organization into subdomains.

scripts/research/

Research-only experiments.

Responsibilities:

calibration experiments
classifier audits
decision-policy audits
reproducible model training
feature importance research

Status:

Active research workspace.

scripts/benchmarks/

Regression and benchmark framework.

Responsibilities:

baseline comparison
schema checks
scoreline regression
calibration regression
benchmark orchestration

Status:

Active benchmark infrastructure.

docs/

Project documentation.

Responsibilities:

research reports
changelog
evaluation notes
architecture documents

Status:

Growing documentation layer.

outputs/

Generated artifacts.

Contains:

benchmark outputs
research outputs
simulation outputs
model-training outputs
versioned baselines

Status:

Generated data; should not be treated as source logic.

Production Flow
1. Historical Dataset

The canonical historical training data is:

outputs/model_training/historical_training_dataset.csv

It contains historical international tournament matches and engineered team-strength features.

2. Expected Goals

Expected goals are produced by the calibrated goal model.

The current production goal engine is:

Dixon-Coles hierarchical sampler
rho = 0.30
tempo_cv = 0.60
team_cv = 0.10
3. Outcome Probabilities

The production outcome model is:

CalibratedClassifierCV
base estimator: LightGBM
calibration: sigmoid
cv = 5
feature set = v2

It predicts:

away win
draw
home win
4. Match Engine

The match engine combines:

expected goals
goal sampling
outcome probabilities
extra time
penalties
5. Tournament Engine

The tournament engine simulates:

group stage
round of 32
round of 16
quarterfinals
semifinals
third-place playoff
final
6. Monte Carlo Engine

The Monte Carlo layer repeatedly simulates tournaments and produces:

champion probabilities
runner-up probabilities
semifinal probabilities
quarterfinal probabilities
round-of-16 probabilities
simulation statistics
Evaluation Framework

The project currently evaluates models across three dimensions.

Scoreline Realism

Primary metric:

total variation distance against historical scoreline distribution

Current baseline:

v5.1_dixon_coles_hierarchical
Probability Calibration

Primary metrics:

multiclass Brier score
multiclass log loss
mean expected calibration error

Diagnostics:

reliability tables
confusion matrix
predicted class distribution
Tournament Behavior

Primary outputs:

advancement probabilities
champion probabilities
simulation statistics
Baseline and Promotion Workflow

Future candidates should follow this process:

Research experiment
       |
       v
Benchmark output
       |
       v
Compare to baseline
       |
       v
Pass regression gates
       |
       v
Promote baseline
       |
       v
Document milestone

Promotion utilities:

scripts/utilities/promote_baseline.py
scripts/utilities/create_baseline_manifest.py

Regression tool:

scripts/benchmarks/compare_to_baseline.py
Current Production Baselines
Version 5.1

Purpose:

scoreline benchmark baseline with explicit production metadata

Location:

outputs/baselines/v5.1_dixon_coles_hierarchical

Status:

Current scoreline benchmark reference.

Research Findings So Far
Goal Model

Dixon-Coles hierarchical sampling improved scoreline realism and is now the recommended production goal engine.

Calibration

Post-processing calibration methods tested:

class multipliers
pseudo-logit temperature scaling

Result:

No production promotion recommended.

The production model remains the best tested probability model.

Decision Policy

Alternative decision rules were tested for draw prediction.

Result:

No production promotion recommended.

The production model rarely assigns draw probabilities high enough to make draw prediction competitive under simple thresholds.

Feature Importance

The production classifier relies heavily on:

goalkeeper difference
FIFA points difference
attack difference
defense difference
Poisson-derived features

This suggests future gains are likely to come from better football features rather than additional calibration.

Current Technical Debt
scripts/ is large

The scripts/ directory contains many workflows and historical experiments.

Future cleanup should gradually separate it into:

scripts/audits/
scripts/data_collection/
scripts/experiments/
scripts/benchmarks/
scripts/research/
scripts/utilities/
Sofascore subsystem

The Sofascore scripts have grown into a data-collection subsystem.

Future target:

data_collection/sofascore/
Historical scripts

Some early scripts may be superseded by newer shared infrastructure.

They should be classified as:

keep
research
archive
review
delete

before removal.

Near-Term Roadmap
Architecture
Complete project architecture documentation.
Classify existing scripts by role.
Avoid creating new top-level scripts unless necessary.
Football Intelligence

Next major project phase should focus on:

richer team-strength modeling
player-stat-derived ratings
improved football-specific features
feature ablation
feature engineering
eventual club and domestic competition support
Competition Expansion

Long-term architecture should support:

domestic leagues
continental club competitions
international club competitions
other international tournaments
Guiding Principles
Production code must stay clean.
Research code must remain isolated.
Benchmark every meaningful change.
Do not promote without regression evidence.
Prefer football-specific features over arbitrary complexity.
Use shared infrastructure instead of duplicated logic.
Document major milestones.