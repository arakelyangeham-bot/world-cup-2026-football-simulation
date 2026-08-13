README.md

# World Cup 2026 Football Simulation Framework

*A modular computational football research platform for simulation, player intelligence, competition modeling, and reproducible football research.*

---

## Overview

This project began as a personal attempt to simulate the 2026 FIFA World Cup using statistical match prediction.

The original goal was straightforward: estimate match scorelines using team strength models and Monte Carlo simulation.

As the project matured, it became clear that accurately modeling football required far more than tournament simulation alone. Improvements to player evaluation, match generation, competition design, and research methodology gradually transformed the project into a broader computational football research platform.

Today, the World Cup remains the flagship application, but the underlying architecture has evolved into a reusable framework for investigating football through reproducible computational experiments.

---

## Project Evolution

The project did not begin with the goal of building a research platform.

It started with a much simpler objective: generate realistic football scoreline predictions and simulate the 2026 FIFA World Cup.

Each stage of development exposed new limitations and raised new questions. Improving match prediction led to player-level modeling. Generalizing tournament logic led to a reusable competition framework. Comparing different modeling choices led to a formal research framework.

The current architecture is therefore the result of incremental refinement rather than a fixed design established at the outset.

---

## Project Philosophy

The project follows several guiding principles:

- **Modularity** — Components should be reusable and independently testable.
- **Research-first development** — New ideas should be evaluated through controlled experiments rather than intuition alone.
- **Reproducibility** — Every experiment should produce structured outputs and documented methodology.
- **Incremental refinement** — Improvements are introduced gradually, with each stage validated before building upon it.

---

## Architecture

The project is organized into several major layers.

### Player Intelligence

Transforms player-level evidence into national team representations.

Key components include:

- PlayerEvidenceRepository
- PlayerRepresentationEngine
- CurrentAbilityEstimator
- TeamRepositoryBuilder
- Multiple repository aggregation strategies

---

### Football Model

Generates realistic football matches.

Components include:

- Production scoreline-first match engine
- Expected goals model
- Goal samplers
- Match engine adapters

---

### Competition Framework

General-purpose competition infrastructure.

Supports:

- League competitions
- Knockout tournaments
- Group stages
- Brackets
- Advancement rules

The World Cup implementation is built on top of these generic abstractions.

---

### Research Framework

Provides infrastructure for computational football experiments.

Features include:

- Experiment definitions
- Experiment conditions
- Metric library
- Experiment runner
- Structured reports
- Reproducible outputs

---

## Research Programs

Current research is organized into thematic programs.

### Competition Research

Investigates how competition structure influences football outcomes.

Completed:

- Experiment 031A — Research Framework Validation
- Experiment 031B — League vs Knockout (Synthetic Football Model)
- Experiment 031C — League vs Knockout (Production Football Model v1)

---

### Football Model Sensitivity

Investigates how football-model components influence research conclusions.

Completed:

- Experiment 032 — Repository Sensitivity Analysis

Planned:

- Experiment 033 — Goal Sampler Sensitivity
- Experiment 034 — Match Engine Sensitivity

---

## Repository Structure

```text
analysis/
outputs/
reports/
research/
simulation/
studies/
```

Each directory corresponds to a distinct layer of the project.

---

## Current Status

The project currently includes:

- Production football model
- Modular competition framework
- Player Intelligence pipeline
- Monte Carlo tournament simulation
- Research framework
- Multiple completed computational football experiments

---

## Long-Term Vision

The long-term objective is to develop an extensible computational football laboratory capable of investigating questions such as:

- How should football competitions be designed?
- Which player aggregation methods best represent team strength?
- Which modeling assumptions most influence tournament predictions?
- How sensitive are football conclusions to changes in the underlying model?

Rather than optimizing solely for prediction accuracy, the project aims to provide a transparent, modular platform for computational football research.

---

## Acknowledgements

This project has been developed as an independent learning and research effort.

It has served as a vehicle for learning software engineering, data science, simulation, statistical modeling, and experimental methodology while exploring the computational aspects of football.