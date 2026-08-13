version_2a_retrospective

# Version 2A Retrospective

## World Cup 2026 Football Simulation Platform

## 1. Purpose

Version 2A marked the transition from a World Cup simulation project into a broader computational football research platform.

The phase was not primarily about producing a better final tournament forecast.

Its purpose was to establish the architecture required to support disciplined football-model development across:

- club competitions;
- national-team competitions;
- historical replay;
- production prediction;
- scoreline simulation;
- scientific evaluation.

Version 2A is therefore best understood as a platform-engineering phase.

---

## 2. Starting point

At the beginning of this phase, the project already contained substantial functionality:

- player-data ingestion;
- team-strength construction;
- expected-goal models;
- multiple scoreline samplers;
- Monte Carlo tournament simulation;
- competition frameworks;
- outcome-classification experiments;
- World Cup-specific utilities.

However, the system still had several architectural uncertainties.

Important questions included:

- whether club football could be represented without Premier League-specific assumptions;
- whether player intelligence could support additional competitions;
- whether research outputs could be promoted into stable production artifacts;
- whether deterministic prediction and stochastic simulation were properly separated;
- whether production prediction had a clear public interface;
- whether replay and evaluation could operate without rerunning or mutating the model;
- whether the broader competition framework could remain independent of model internals.

Version 2A addressed these questions systematically.

---

## 3. Major architectural achievements

### 3.1 Competition-generalized football intelligence

The player-intelligence architecture was extended from narrow competition workflows into reusable competition-aware components.

The resulting chain became:

```text
Processed player evidence
        ↓
CompetitionPlayerRepository
        ↓
CompetitionRosterBuilder
        ↓
CompetitionTeamRepository
        ↓
TeamRepresentation

This proved that the core football-intelligence architecture was not inherently tied to one league.

The Bundesliga integration served as the decisive validation case.

3.2 Production repository architecture

A dedicated production repository layer was introduced.

The central pattern became:

Research-domain TeamRepresentation
        ↓
ProductionClubRecord
        ↓
ProductionClubRepositoryBuilder
        ↓
Versioned persisted repository

This separated:

football-intelligence calculation;
persistence;
runtime loading.

That distinction is fundamental.

The runtime no longer needs to reconstruct player intelligence every time a fixture is predicted.

3.3 Date-aware rating priors

ClubElo integration established a reusable mechanism for resolving external club-strength priors at the correct prediction date.

The integration validated:

cache persistence;
name resolution;
temporal intervals;
runtime lookup;
competition independence.

The Holstein Kiel alias issue also demonstrated why external-provider identities must be treated as explicit data contracts rather than assumed name matches.

3.4 Live observation construction

The LiveMatchObservationBuilder established a stable boundary between football intelligence and model inference.

Its role is narrowly defined:

Production club representation
        +
Date-valid rating prior
        ↓
Model-ready live observation

It does not fit models.

It does not simulate matches.

It does not rebuild player representations.

This boundary significantly reduced hidden coupling.

3.5 Frozen production goal model

The project established a proper frozen expected-goal artifact and runtime wrapper.

The production model:

loads from a persisted artifact;
validates its own contract;
exposes required features;
returns expected home and away goals;
preserves training and artifact provenance;
performs no fitting at prediction time.

This completed the separation between model development and model deployment.

3.6 Authoritative prediction API

The ProductionPredictionPipeline became the authoritative deterministic club-fixture prediction interface.

Its runtime sequence is:

Fixture request
        ↓
LiveMatchObservation
        ↓
ProductionGoalModel
        ↓
lambda_home and lambda_away
        ↓
Outcome probabilities
        ↓
Immutable prediction record

The pipeline provides a complete prediction object containing:

fixture identity;
feature values;
rating priors;
expected goals;
outcome probabilities;
repository provenance;
model provenance.

This was one of the most important achievements of Version 2A.

Before this phase, prediction responsibilities existed across multiple modules.

After Version 2A, the public club prediction pathway became explicit.

3.7 Preservation of scoreline-first simulation

The architecture review confirmed that the active simulation philosophy remains scoreline-first.

The causal sequence is:

Expected goals
        ↓
Scoreline distribution
        ↓
Sampled scoreline
        ↓
Derived result
        ↓
Competition progression

The current default configuration uses the scoreline-first path.

A legacy outcome-first route remains available for comparison and compatibility, but it is not the default.

This is more than an implementation detail.

It is one of the core modeling principles of the project.

3.8 Generic football-model boundary

The review clarified that the most important simulation abstraction is not one particular prediction pipeline.

It is the football-model interface:

simulate_match(
    home_team,
    away_team,
    prediction_date=...
)

Competition code consumes scorelines through this interface.

It does not need to understand:

ClubElo;
player repositories;
national-team ratings;
feature vectors;
model artifacts;
Poisson regression;
scoreline-sampler internals.

This dependency inversion allows football models and competition engines to evolve independently.

3.9 End-to-end club simulation

Studies 073 and 074 validated that the club production runtime was already connected to the broader competition framework.

The complete pathway was demonstrated as:

Production repository
        ↓
Live observation builder
        ↓
Integrated club goal model
        ↓
Expected goals
        ↓
Existing scoreline sampler
        ↓
LeagueMatchSimulator
        ↓
CompetitionEngine

This validation included:

fixture dates;
ClubElo resolution;
scoreline sampling;
complete fixture populations;
standings;
competition arithmetic.

The architecture review prevented unnecessary rebuilding of this bridge.

3.10 Replay and evaluation separation

Version 2A established a disciplined distinction among:

Prediction generation
        ↓
Replay artifact
        ↓
Performance analysis

Study 083 generated the replay artifact.

Study 084 evaluated only that frozen artifact.

This ensured that evaluation did not:

rerun predictions;
alter runtime configuration;
refit the model;
silently change inputs.

This separation materially improved the scientific credibility of the platform.

4. Important research discoveries

Version 2A was primarily architectural, but it also produced meaningful football-model findings.

4.1 Stable cross-competition operation

The Bundesliga integration showed that the architecture could support another major domestic league without redesigning core components.

This was stronger evidence than a theoretical claim of generality.

The system successfully handled:

competition evidence;
club representations;
production repositories;
ClubElo histories;
live observations;
fixture predictions;
full-season replay.
4.2 Strong draw-rate behavior

The Bundesliga replay diagnostic found close agreement between actual and predicted draw rates.

This suggests that earlier work on draw behavior and low-score structure produced a meaningful improvement.

It also showed why evaluation must examine more than simple outcome accuracy.

4.3 Systematic goal-volume underprediction

The most important modeling finding was systematic underprediction of Bundesliga scoring.

Observed average total goals were materially higher than predicted total goals.

This indicates a structured model bias rather than purely random fixture-level error.

Possible explanations include:

league scoring-environment differences;
compression of team-strength representations;
insufficient elite attacking differentiation;
static season-level representations;
missing lineup information;
model regularization;
omitted tactical or availability context.

This finding provides a concrete starting point for Version 2B.

4.4 Team-level bias

The replay analysis showed notable team-specific error patterns.

Bayern was substantially underestimated in goal difference, while Bochum was relatively overrated.

These patterns suggest that aggregate calibration alone is insufficient.

Future evaluation must examine whether errors arise from:

attack estimates;
defensive estimates;
player aggregation;
rating priors;
interactions between representation and model coefficients.
5. What the architecture review corrected

The review was valuable partly because it corrected several assumptions.

5.1 Production and studies were less coupled than expected

The production runtime did not depend directly on Studies 078–084.

Studies created and validated artifacts, while shared modules handled runtime behavior.

This confirmed successful study-to-framework promotion.

5.2 The club simulation bridge already existed

At one point, it appeared that a new integration study might be needed to connect the production pipeline to league simulation.

Inspection of Studies 073 and 074 showed that this bridge had already been built and validated.

The review prevented duplicated work.

5.3 Club and national-team runtimes should remain distinct

The club production pipeline and World Cup runtime share scoreline-first philosophy, but they operate in different football domains.

The club runtime uses:

club representations;
ClubElo;
a club-trained production artifact.

The World Cup runtime uses:

national-team repositories;
national-team rating priors;
national-team calibration.

Architectural consistency does not require forcing both domains through one identical feature-generation pipeline.

The correct common abstraction is the football-model interface.

5.4 Historical accumulation is not the same as structural failure

The repository contains many older:

inference modules;
outcome classifiers;
model experiments;
analysis scripts;
samplers;
compatibility routes.

The review found that the main issue is lifecycle clarity, not broken architecture.

These components should be classified and documented rather than deleted casually.

6. What worked well during Version 2A

Several working principles were especially effective.

6.1 Inventory before redesign

The project repeatedly benefited from inspecting existing code before proposing new modules.

This prevented reinvention and exposed already-completed integration work.

6.2 Promote reusable logic

When a study produced functionality with general value, that functionality was moved into shared modules.

Examples include:

production repository construction;
scoreline probability calculation;
production prediction pipeline;
runtime factory construction.

This kept study scripts from becoming hidden production dependencies.

6.3 Validate one boundary at a time

The progression from Studies 078 through 084 was deliberately staged.

Each study answered a bounded question:

can the repository be built?
can ClubElo resolve?
can observations be assembled?
can the model predict?
are distributions sane?
can the pipeline operate as one API?
can a season be replayed?
what does performance reveal?

This sequence made failures easier to localize.

6.4 Preserve artifacts and provenance

Outputs were treated as scientific records rather than disposable files.

Versioned repositories, model artifacts, replay outputs, metadata and reports now provide an auditable chain from evidence to evaluation.

6.5 Separate operational success from predictive success

The project explicitly distinguished:

Pipeline executed correctly

from:

Model predicted football well

Study 083 established operational replay success.

Study 084 evaluated predictive behavior.

This distinction should remain permanent.

7. Known limitations at the freeze boundary

Version 2A is complete, but it is not perfect.

7.1 Public API hierarchy

The project contains:

ProductionPredictionPipeline
ProductionGoalModel
IntegratedClubGoalPredictor

Their hierarchy is now understood, but it should remain clearly documented.

The preferred external prediction entry point is:

ProductionPredictionPipeline
7.2 Legacy lifecycle labeling

Older prediction and classification stacks still require explicit lifecycle labels.

A future cleanup may classify modules as:

active;
compatibility;
research-only;
deprecated;
archived.

This is documentation and maintenance work, not a Version 2A blocker.

7.3 Club identity mappings

External provider name overrides remain distributed through runtime mappings.

A centralized provider-aware identity registry may eventually improve maintainability.

7.4 Static team representations

Current production repositories are effectively frozen team snapshots.

They do not yet fully capture:

transfers during the evaluation period;
injuries;
suspensions;
rotation;
expected starting lineups;
short-term form;
tactical changes.

This is likely one of the most important scientific limitations.

7.5 Incomplete representation metadata

Some fields, including squad_quality and evidence_score, were zero in the Bundesliga production repository.

These fields were not used by the current goal-model contract, but their condition suggests unfinished upstream intelligence behavior.

7.6 Evaluation overlap

The Bundesliga replay season overlapped the production model’s training period.

Study 084 therefore provides a diagnostic rather than a clean estimate of prospective generalization.

A proper forward-period or cross-season test remains necessary.

8. Version 2B research agenda

Version 2B should not begin with another broad architectural expansion.

It should begin with targeted scientific questions derived from observed model behavior.

The most important question is:

Why does the production model systematically underpredict goals, especially for certain teams?

Candidate research directions include:

8.1 Scoring-environment calibration

Determine whether leagues require explicit scoring-environment adjustments.

Possible mechanisms include:

league intercepts;
season intercepts;
hierarchical environment effects;
calibrated lambda scaling;
league-specific variance structures.

This must be studied carefully to avoid masking deeper representation problems.

8.2 Expected starting lineups

Replace static full-squad representations with fixture-specific expected-XI representations.

This could improve sensitivity to:

player availability;
squad rotation;
attacking concentration;
goalkeeper changes;
injuries;
transfers.

The project already contains foundational lineup work that can be integrated rather than reinvented.

8.3 Transfer-aware repositories

Build representations whose membership and evidence are valid for the prediction date.

This is especially important for historical replay.

A club’s end-of-season squad should not be used unchanged for matches played before transfers occurred.

8.4 Dynamic form

Assess whether existing dynamic-form research should be promoted into the production observation contract.

This should occur only after confirming incremental out-of-sample value.

8.5 Elite-team nonlinearities

Investigate whether the model compresses the strongest attacks and defenses.

Bayern’s replay bias may indicate that linear effects or current aggregation methods fail at the upper tail.

Possible research questions include:

whether attacking strength requires nonlinear terms;
whether star-player concentration matters;
whether attack depth behaves differently for elite clubs;
whether rating priors and player intelligence overlap imperfectly.
8.6 Clean generalization testing

Establish an evaluation period that occurs strictly after:

model fitting;
repository construction;
representation cutoff;
rating-prior availability.

This is necessary before making strong claims about production forecasting quality.

8.7 National-team production modernization

The World Cup runtime remains valid and scoreline-first, but it does not yet possess the same complete production artifact and replay architecture as the club runtime.

A future national-team production phase may adapt the lessons from Version 2A without assuming club and national-team football are identical domains.

9. Freeze principles

The following boundaries should be preserved during Version 2B.

Principle 1

Football intelligence should not be calculated inside the scoreline sampler.

Principle 2

Competition engines should not depend on model-specific features.

Principle 3

Production inference should not refit models.

Principle 4

Prediction generation and evaluation should remain separate.

Principle 5

Scorelines should remain primary simulation realizations.

Principle 6

Reusable study logic should be promoted into shared modules.

Principle 7

Architectural changes should be justified by observed boundary failure, not convenience alone.

10. Final retrospective

Version 2A succeeded because it did not attempt to solve every football-modeling problem at once.

Instead, it created the conditions under which those problems can now be studied properly.

The platform now supports a complete scientific cycle:

Evidence
        ↓
Representation
        ↓
Production artifact
        ↓
Prediction
        ↓
Scoreline simulation
        ↓
Competition
        ↓
Replay
        ↓
Evaluation
        ↓
Research hypothesis

That final transition is the most important.

The project no longer merely produces tournament simulations.

It now generates falsifiable questions about football modeling.

Version 2A therefore closes with the following status:

ARCHITECTURE: COMPLETE
PRODUCTION FREEZE: APPROVED
NEXT PHASE: VERSION 2B — SCIENTIFIC MODEL IMPROVEMENT