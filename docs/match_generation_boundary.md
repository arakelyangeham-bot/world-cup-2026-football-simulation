match_generation_boundary.md

# Match Generation Boundary

## Purpose

This document defines the architectural boundary between the football model and the competition engine.

## Core Contract

The competition engine should depend on one thing:

```python
simulate_match_score(team1_data, team2_data) -> tuple[int, int]