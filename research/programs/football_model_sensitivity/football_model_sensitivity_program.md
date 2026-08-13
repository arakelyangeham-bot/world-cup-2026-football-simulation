# Football Model Sensitivity Program

## Mission

Competition Research Program A demonstrated that competition format influences tournament outcomes.

It also showed that replacing the synthetic football model with Production Football Model v1 changed the magnitude of several results while preserving the qualitative conclusion.

The purpose of the Football Model Sensitivity Program is to determine which components of the football model materially influence research conclusions.

## Core Question

How sensitive are computational football conclusions to the choice of football model?

## Philosophy

Each experiment should change exactly one football-model component while keeping all other variables fixed.

## Planned Experiments

### Experiment 032 — Repository Sensitivity Analysis

Independent variable:

- `legacy`
- `dimension_specific`
- `top_11_mean`
- `top_5_mean`
- `star_weighted`
- `starter_plus_depth`

Question:

> How sensitive are competition outcomes to the choice of team representation repository?

### Experiment 033 — Goal Sampler Sensitivity

Independent variable:

- production Dixon-Coles hierarchical sampler
- simpler Poisson sampler
- future goal samplers

Question:

> How sensitive are results to scoreline sampling assumptions?

### Experiment 034 — Match Engine Sensitivity

Independent variable:

- synthetic match model
- production scoreline-first engine
- future match engines

Question:

> How sensitive are competition conclusions to the match engine itself?

## Methodological Principle

A football-model sensitivity experiment should not change the research question, metric definitions, competition format, or simulation count unless explicitly justified.

## Strategic Importance

This program turns Version 2 modeling choices into Version 3 scientific variables.

The goal is no longer simply to choose a preferred model, but to understand which modeling decisions materially affect football research conclusions.