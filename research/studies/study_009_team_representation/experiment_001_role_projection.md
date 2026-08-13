experiment_001_role_projection.md

# Experiment 001 — Role Projection

## Purpose

Evaluate how role-specific player ratings should be projected into team-level representation dimensions.

## Current Pipeline

```text
Player Ratings
    ↓
Role Ratings
    ↓
Role Projection
    ↓
Team Representation
    ↓
Team Repository Entry

Current Projection

Attack:

ST 40%
W 25%
AM 20%
CM 10%
FB 5%

Midfield:

CM 35%
DM 25%
AM 20%
WM 10%
FB 10%

Defense:

CB 40%
FB 25%
DM 25%
GK 10%

Goalkeeper:

GK 100%
Research Question

Which role projection best preserves meaningful football strength at the team level?

Hypothesis

The current projection is structurally reasonable, but the resulting team representation may need rescaling before it can serve as a production Team Repository input.

Metrics

Evaluate each projection by:

dynamic range
standard deviation
separation between strong and weak teams
compatibility with expected-goals model
downstream scoreline realism
Candidate Projection Families
Projection A — Current

Use current hand-designed role weights.

Projection B — Attack-heavy

Increase the contribution of attacking roles to attack.

Projection C — Midfield-control

Increase central midfield and defensive midfield influence.

Projection D — Defense-core

Increase CB, DM, and GK influence in defensive representation.

Projection E — Starting-XI weighted

Use only expected starters instead of full roster.

Non-Goals

This experiment will not modify:

match engine
goal sampler
tournament simulator
Poisson model coefficients
Success Criteria

A projection candidate is promising if it:

produces meaningful variation across teams,
remains interpretable,
avoids collapsing all teams into a narrow range,
can be benchmarked through the existing scoreline calibration framework.
Next Step

Implement a role projection comparison script that builds team representations for all teams under multiple projection variants.