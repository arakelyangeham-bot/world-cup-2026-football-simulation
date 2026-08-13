research_protocol.md

# Research Protocol 011

# Team Representation Calibration

## Purpose

Study how player-derived football knowledge should be transformed into team-level simulation inputs.

## Motivation

Player Intelligence v1 can now build a complete team repository in parallel with the legacy production path.

Comparison against the legacy repository showed that the Player Intelligence representation is structurally valid but dynamically compressed.

The next problem is no longer architecture. It is aggregation.

## Primary Research Question

How should a collection of players be transformed into a team representation suitable for football simulation?

## Core Problem

Player Intelligence currently provides a faithful player-derived representation.

The simulator requires a discriminative team representation.

This study investigates how to bridge that gap.

## Candidate Aggregation Strategies

### Strategy A — Current Top-N Mean

Use the existing top-N mean representation.

### Strategy B — Expected XI Only

Use expected starters only.

### Strategy C — Expected XI + Bench Depth

Weight starters strongly and bench players lightly.

### Strategy D — Elite-Player Weighting

Allow star players to influence team strength disproportionately.

### Strategy E — Position-Balance Weighting

Preserve balance across goalkeeper, defense, midfield, and attack.

## Evaluation Metrics

- dynamic range
- standard deviation
- team separation
- overlap with legacy repository
- scoreline realism
- tournament realism

## Non-Goals

This study will not change:

- match engine
- goal sampler
- tournament simulator
- Poisson coefficients

## Deliverables

- aggregation strategy comparison
- recommended aggregation approach
- calibrated Player Intelligence team repository
- production integration recommendation

## Guiding Principle

Do not tune the simulator to compensate for weak team representation.

Improve the player-to-team transformation first.