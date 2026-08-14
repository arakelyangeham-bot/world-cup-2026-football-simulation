# World Cup 2026 Football Simulation Framework

*A modular computational football research platform for player intelligence, match prediction, football simulation, competition modeling, and reproducible experimentation.*

---

## Overview

This project began as a personal attempt to simulate the 2026 FIFA World Cup using statistical match prediction.

The original objective was relatively simple: estimate football scorelines from team strength and use Monte Carlo simulation to model the tournament.

As development continued, each improvement exposed a deeper modeling problem. Team-level prediction led to player-level evaluation. Player evaluation led to questions about evidence, identity, roles, expected lineups, and team representation. Tournament simulation led to reusable competition infrastructure. Comparing alternative modeling choices led to a formal research and validation framework.

The World Cup remains the flagship application, but the project has evolved into a broader computational football research platform.

The emphasis is not only on producing predictions. It is on understanding how football representations, modeling assumptions, data choices, and competition structures influence those predictions.

---

## Development Philosophy

The project is developed around several principles:

* **Modularity** — components should have clear responsibilities and reusable interfaces.
* **Incremental development** — validated behavior is preserved while new ideas are introduced one component at a time.
* **Research before promotion** — alternative approaches are evaluated experimentally before entering production-facing architecture.
* **Empirical validation** — architectural changes should be supported by measurable evidence rather than intuition alone.
* **Provenance awareness** — source identity, data grain, temporal context, and evidence lineage are treated as modeling concerns.
* **Reproducibility** — model-defining configuration, research methodology, validation logic, and important mappings are version controlled.
* **Separation of concerns** — data acquisition, Player Intelligence, prediction, simulation, competition logic, and research infrastructure remain conceptually distinct.

The repository intentionally contains both production-facing infrastructure and the research history used to justify many of its design decisions.

---

## Major Architecture

### Player Intelligence

The Player Intelligence pipeline transforms historical player evidence into canonical player features, attributes, and role ratings.

A key architectural rule is that historical evidence remains at:

```text
competition × season × canonical player
```

until historical evidence weighting has occurred.

The modern production pipeline is:

```text
Raw player statistics
        ↓
Canonical evidence resolution
        ↓
Competition-season feature engineering
        ↓
Historical evidence weighting
        ↓
Canonical player features
        ↓
Player attributes
        ↓
Role ratings
```

Player identity is maintained separately through a canonical player registry built from current player profiles.

The authoritative Player Intelligence entry point is:

```bash
python -m scripts.run_player_intelligence_pipeline
```

More detail is available in:

```text
docs/player_intelligence_pipeline.md
```

---

### Team Representation

Player-level information can be transformed into team-level representations through multiple aggregation and lineup strategies.

Research in this area includes:

* expected-lineup construction;
* role-sensitive player evaluation;
* depth and contribution modeling;
* alternative aggregation families;
* representation calibration;
* contextual player contribution;
* formation geometry;
* positional and structural responsibility modeling.

Research implementations primarily live under:

```text
research/player_intelligence/
research/studies/
```

Validated production-facing team representations are consumed downstream through stable repository interfaces.

---

### Match Prediction and Goal Modeling

The project contains multiple football-modeling layers, including:

* expected-goals estimation;
* scoreline-first match generation;
* Poisson-based goal modeling;
* calibrated and hierarchical goal samplers;
* rating-prior integration;
* production prediction interfaces;
* probability calculation and calibration;
* historical replay and benchmarking infrastructure.

Relevant runtime code is distributed across:

```text
inference/
simulation/
research/production/
```

The repository also preserves earlier modeling approaches and experimental alternatives for comparison and historical reproducibility.

---

### Competition Framework

Competition logic is modeled independently from the football model that produces individual match outcomes.

The framework supports concepts including:

* league competitions;
* group stages;
* knockout rounds;
* standings;
* brackets;
* advancement rules;
* multi-stage competitions;
* fixture generation.

Core competition infrastructure lives primarily under:

```text
simulation/competition/
competition_catalog/
fixture_generation/
```

The 2026 FIFA World Cup implementation is built on top of this broader competition architecture.

---

### Tournament Simulation

The World Cup simulation layer combines:

```text
Team representation
        ↓
Match prediction / scoreline generation
        ↓
Group-stage simulation
        ↓
Advancement
        ↓
Knockout bracket
        ↓
Monte Carlo tournament simulation
```

Repeated simulations are used to estimate stage and tournament probabilities rather than treating any single simulated tournament as a prediction.

---

## Research Framework

A substantial part of this repository is devoted to controlled football research rather than production execution.

Research includes:

* representation studies;
* goal-model benchmarking;
* calibration experiments;
* competition research;
* historical replay;
* feature ablation;
* player-selection studies;
* aggregation mathematics;
* formation and structural-responsibility studies;
* football-evidence and observability investigations.

Research code and study definitions primarily live under:

```text
research/
```

Generated study outputs are generally **not** committed to Git.

Where a study produces a lasting architectural decision, the repository may retain methodology, decision documents, freeze notes, validation code, or small reproducibility artifacts.

Research code should therefore not automatically be interpreted as the currently selected production architecture.

---

## Repository Structure

The repository is intentionally modular. Major top-level areas include:

```text
analysis/               model analysis and diagnostic utilities

benchmarks/             benchmark documentation and reference material

competition_catalog/    reusable competition definitions and builders

data/                   selected configuration, mappings, and reference inputs

docs/                   architecture, methodology, and production documentation

experiments/            experimental infrastructure and earlier experiment tooling

fixture_generation/     generic fixture-generation logic

inference/              prediction and probability interfaces

models/                 machine-learning model implementations

research/               formal studies, research infrastructure, prototypes,
                        benchmarking, and production-candidate evaluation

scripts/                operational pipelines, builders, audits, validation,
                        ingestion, and World Cup workflows

shared/                 reusable cross-cutting utilities and configuration

simulation/             match, goal, competition, and tournament simulation

tests/                  automated test suite

tuning/                 model hyperparameter tuning

validation/             model and data validation
```

Not every directory represents a current production runtime path. Some preserve earlier experiments or alternative modeling approaches intentionally.

---

## Installation

The project is currently developed and tested with:

```text
Python 3.11
```

Create a virtual environment and install the direct Python dependencies:

```bash
python -m venv .venv
```

Then install:

```bash
python -m pip install -r requirements.txt
```

The dependency specification has been validated in a clean Python 3.11 virtual environment.

Some acquisition workflows use Playwright and may require installation of its browser runtime separately.

---

## Data and Reproducibility

The repository is designed to be **source-reconstructible rather than self-contained**.

Large scraped datasets, generated production artifacts, model outputs, research datasets, and temporary analysis products are intentionally excluded from Git.

Examples of excluded material include:

```text
data/processed/
outputs/
research/data/raw/
research/data/processed/
large raw Sofascore datasets
generated interactive reports
```

Small artifacts that define model behavior or are important for reproducibility remain version controlled. Examples include:

```text
competition_manifest.csv
competition_feature_manifest.csv
feature_attribute_manifest.csv
role_attribute_manifest.csv
role_feature_manifest.csv
formation_manifest.csv
stat_manifest.csv
sofascore_player_id_aliases.csv
formation_geometry.csv
```

Raw football data can generally be reconstructed through the repository's ingestion and discovery scripts.

For example, the Player Intelligence acquisition path begins with competition configuration and discovery, then builds player membership, profiles, and competition-season statistics before canonical processing begins.

Because acquisition depends on external data providers, exact reconstruction is not guaranteed indefinitely. External endpoints, schemas, availability, rate limits, and source data may change over time.

The repository therefore distinguishes between:

```text
version-controlled methodology and configuration
        ↓
external data acquisition
        ↓
local raw artifacts
        ↓
validated transformations
        ↓
local generated outputs
```

---

## Testing and Validation

The project includes an automated test suite under:

```text
tests/
```

Testing covers areas including:

* Player Intelligence aggregation;
* historical compatibility;
* player contribution;
* role suitability;
* formation geometry;
* structural responsibility;
* team-representation integration;
* competition infrastructure;
* prediction interfaces.

The project also uses dedicated validation scripts and empirical research studies where conventional unit tests are not sufficient to evaluate football-model behavior.

---

## Documentation

Important documentation includes:

```text
docs/player_intelligence_pipeline.md
    Modern Player Intelligence production architecture

docs/architecture/
    Project and subsystem architecture documentation

research/
    Research methodology, study implementations, and experimental infrastructure

research/studies/study_101_competition_expansion/STUDY_101F_FREEZE.md
    Study 101F architecture decision and production freeze
```

The documentation directory also preserves earlier architecture and model-development notes. Some documents describe historical states rather than the current production architecture.

---

## Current Status

The project currently contains working infrastructure for:

* player-data ingestion;
* canonical player identity and evidence resolution;
* historical Player Intelligence;
* player attributes and role ratings;
* expected-lineup and team-representation research;
* match prediction;
* scoreline-first match simulation;
* calibrated goal modeling;
* generic competition simulation;
* World Cup group and knockout architecture;
* Monte Carlo tournament simulation;
* historical benchmarking and replay;
* formal football research and validation.

The project remains under active development.

It should be viewed as an evolving research platform rather than a finished predictive product.

---

## Limitations

Football prediction is inherently uncertain, and this project does not attempt to eliminate that uncertainty.

Current limitations include:

* dependence on externally sourced football data;
* incomplete or changing player-data coverage;
* uncertainty in future squads and expected lineups;
* simplified representations of tactical and match context;
* model assumptions that may behave differently across competitions and eras;
* research components that have not necessarily been promoted into production;
* ongoing architectural evolution.

Simulation probabilities should therefore be interpreted as outputs of the current model and data assumptions, not as objective forecasts of future football outcomes.

---

## Long-Term Direction

The long-term goal is to build an extensible computational football laboratory for questions such as:

* How should player evidence be transformed into team strength?
* Which aggregation strategies best capture football squad quality?
* How do formation and positional structure change team representation?
* Which match-generation assumptions most strongly influence predictions?
* How robust are model conclusions across competitions and seasons?
* How does competition format affect tournament outcomes?
* Which football phenomena are observable with the available data?

The World Cup remains the central application, but the architecture is intended to support broader football research.

---

## Acknowledgements

This project has been developed as an independent learning and research effort.

It began with player statistics and World Cup simulation and gradually expanded into a vehicle for learning and applying software engineering, data science, statistical modeling, simulation, experimental design, and football analytics.
