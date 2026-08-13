project_architecture.md

# World Cup 2026 Data Science Project Architecture

## Purpose

This project builds a modular football prediction and tournament simulation platform.

The core goal is to transform football data into match predictions, scoreline simulations, and full competition simulations.

Long-term target domains include:

- FIFA World Cup simulation
- international tournaments
- domestic leagues
- international club competitions
- player-stat-driven team strength modeling

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