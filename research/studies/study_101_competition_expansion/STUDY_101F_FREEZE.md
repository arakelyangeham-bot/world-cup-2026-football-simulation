STUDY_101F_FREEZE.md

# Study 101F — Canonical Historical Evidence and Weighted Player Representation

## Status

**FROZEN — VALIDATED AND PROMOTED TO PRODUCTION**

Study 101F established the production architecture for canonical
historical player evidence, competition-season weighting, and
player-level feature representation.

The final architecture was validated experimentally and subsequently
promoted into the production Player Intelligence pipeline.

---

## 1. Motivation

Study 101 expanded the Player Intelligence evidence population across
additional competitions and seasons.

During that expansion, an architectural problem became visible:
historical player evidence was being collapsed to one row per source
`player_id` before all competition-season provenance had served its
modeling purpose.

This created two related risks:

1. competition and season provenance could be destroyed before
   historical weighting;
2. multiple provider IDs representing the same footballer could be
   treated as separate player identities.

Study 101F was created to resolve these issues without changing
unrelated Player Intelligence mathematics.

---

## 2. Canonical Identity Case

A concrete identity case exposed the problem.

Neliso Senzo Dlamini appeared in historical evidence under two
Sofascore player IDs:

```text
990970
1218855

The reviewed canonical identity is:

canonical_player_id = 1218855

The two source observations associated with the canonical collision
were verified to contain equivalent football evidence.

The canonical evidence resolver therefore:

preserves the original source player_id;
assigns the reviewed canonical_player_id;
detects duplicate canonical competition-season task keys;
verifies collision equivalence;
requires a native canonical source observation;
retains the native canonical row deterministically.

This changed the frozen Study 101 evidence population from:

36,643 source evidence rows
14,930 source player IDs

to:

36,642 canonical evidence rows
14,929 canonical footballers

without losing a genuine football observation.

3. Competition-Season Provenance Defect

The pre-101F production path historically performed approximately:

competition-season statistics
        |
        v
historical player aggregation
        |
        v
one row per source player
        |
        v
feature engineering
        |
        v
attribute scoring

Competition and season information was therefore collapsed too early.

Later stages could attempt to apply competition or recency weighting,
but the original evidence scopes were no longer available.

Study 101F demonstrated this directly through competition-season
manifest coverage.

Before the provenance repair, only approximately:

50.79%

of the relevant feature rows could be validly associated with their
competition-season weighting scope.

After preserving competition-season evidence until weighting:

100%

of the canonical feature rows had valid competition-season joins.

4. Canonical Evidence Grain

Study 101F established the historical evidence key:

competition_id
× season_id
× canonical_player_id

Canonicalization occurs before historical evidence aggregation.

The source player_id remains provenance and is not overwritten.

A duplicate canonical task key is not automatically discarded.
Canonical collisions must first be shown to contain equivalent
football evidence.

5. Historical Evidence Weighting

Historical evidence is weighted before player-level feature collapse.

The weighting system incorporates:

minutes played
× season recency
× competition importance

and feature-level competition availability.

Feature availability is applied before aggregation so that a
competition-season observation does not contribute to a feature when
that feature was unavailable for the relevant evidence scope.

The resulting representation contains one row per canonical player.

6. Rejected Intermediate Architecture

An intermediate 101F implementation applied robust feature
normalization at competition-season evidence grain before historical
aggregation.

That architecture produced substantial rating movement, including
unexpected movement among players with only one historical evidence
scope.

The initial comparison produced approximately:

Population	Mean absolute role-rating change	Median	P90
Single-scope	0.059046	0.005611	0.164747
Multi-scope	0.426517	0.164912	1.207835

The single-scope population was intended to function as a control:
players with only one evidence scope should not experience large
representation changes merely because historical weighting had been
repaired.

Investigation showed that the intermediate implementation had also
changed the normalization boundary.

This was not necessary to solve the provenance problem.

The evidence-grain normalization architecture was therefore rejected.

7. Preserved Study 094 Representation Boundary

The final architecture preserves robust normalization at player grain.

The ordering is:

canonical competition-season evidence
        |
        v
per-90 raw features
        |
        v
competition / recency / availability weighting
        |
        v
weighted RAW canonical-player features
        |
        v
robust_zscore at PLAYER GRAIN
        |
        v
Player Intelligence attributes
        |
        v
role ratings

This preserves the previously validated robust player-representation
boundary while correcting historical evidence weighting upstream.

8. Final Rating-Impact Validation

The final 101F architecture was compared against the preceding 101E
ratings across 14,929 matched canonical players.

Evidence-scope populations:

single-scope players: 6,728
multi-scope players:  8,201

Players with comparable role ratings:

all matched:   8,174
single-scope:  2,891
multi-scope:   5,283

Final rating-impact summary:

Population	Mean absolute role-rating change	Median	P90	Maximum
All matched	0.085625	0.023471	0.228566	3.665752
Single-scope	0.006782	0.002563	0.016000	0.677420
Multi-scope	0.128770	0.062409	0.316802	3.665752

The final architecture therefore produced the intended validation
signature:

single-scope population
        -> highly stable


multi-scope population
        -> materially larger movement

Mean absolute movement among multi-scope players was approximately
19 times larger than among single-scope players.

The top overall movers were concentrated in the multi-scope
population.

This supports the interpretation that 101F primarily changed players
whose historical representation genuinely depended on combining
multiple competition-season evidence scopes.

9. Sparse Robust-Z Behavior

During Study 101F, some sparse features were observed to produce large
robust-z values when the population median absolute deviation was very
small.

This behavior was reproducible at player grain and predated the
competition-provenance repair.

It was therefore classified as a separate future
transformation-calibration research question rather than a defect
introduced by Study 101F.

No transformation change was made as part of the 101F freeze.

10. Production Promotion

The validated research architecture was promoted incrementally into
production components.

Production components:

scripts/resolve_player_evidence.py
scripts/sofascore_feature_engineering.py
scripts/build_canonical_player_registry.py
scripts/build_weighted_player_features.py
scripts/score_player_attributes.py
scripts/build_player_ratings_v4.py

A dedicated orchestrator was added:

scripts/run_player_intelligence_pipeline.py

The old general Sofascore orchestrator is not the authoritative
Player Intelligence runner.

11. Numerical Promotion Tests

Before production-default wiring, each major promoted representation
was compared directly against its validated research counterpart.

Weighted player features
Research rows:              14,929
Production rows:            14,929
Columns equal:              True
Canonical IDs equal:        True
Maximum numeric difference: 0
Player attributes
Research rows:              14,929
Production rows:            14,929
Columns equal:              True
Canonical IDs equal:        True
Maximum numeric difference: 0
Player ratings
Research rows:              14,929
Production rows:            14,929
Columns equal:              True
Canonical IDs equal:        True
Maximum numeric difference: 0
Canonical registry promotion

The source-registry-based production implementation initially
reproduced the frozen research canonical registry exactly:

Research rows:         14,929
Production rows:       14,929
Columns equal:         True
Canonical IDs equal:   True
Entire artifact equal: True

The final production identity branch was subsequently simplified to
build the canonical registry directly from the current player-profile
population.

The current profile population and frozen canonical-registry
population contain the same 14,929 canonical IDs.

12. Final Production Replay

The six-stage production Player Intelligence pipeline completed
successfully from its normal default paths.

Final production audit:

Canonical evidence rows:       19,382
Canonical evidence players:     9,698
Duplicate evidence task keys:        0


Feature rows:                  19,382
Feature players:                9,698
Feature task keys unique:        True


Weighted feature rows:          9,698
Weighted canonical IDs:         9,698
Weighted duplicate IDs:             0


Canonical registry rows:       14,929
Registry canonical IDs:        14,929
Registry duplicate IDs:             0


Attribute rows:                 9,698
Attribute canonical IDs:        9,698
Attribute duplicate IDs:            0


Rating rows:                    9,698
Rating canonical IDs:           9,698
Rating duplicate IDs:               0

Cross-stage invariants:

weighted population == attribute population   True
attribute population == rating population     True
all rated IDs exist in canonical registry     True

The canonical registry contained:

5,231

players without current weighted statistical evidence.

This is treated as a data-coverage property rather than pipeline data
loss.

13. Legacy Architecture

The following components remain available for historical,
general-purpose, or research workflows but are not part of the modern
Player Intelligence production path:

aggregate_player_history.py
aggregate_player_history_v2.py
wc_2026_player_dataset.csv
build_player_registry.py
run_sofascore_pipeline.py

Historical studies that reference these components remain valid
records of earlier project architecture.

They should not be interpreted as the current production run order.

14. Final Decision

Study 101F is frozen.

The production Player Intelligence architecture is:

source competition-season evidence
        |
        v
canonical identity resolution
        |
        v
canonical competition-season evidence
        |
        v
per-90 feature engineering
        |
        v
competition / recency / availability weighting
        |
        v
weighted raw canonical-player representation
        |
        v
robust player-grain transformation
        |
        v
Player Intelligence attributes
        |
        v
role-specific player ratings

Current player profiles form a separate canonical identity branch and
provide the authoritative registry used downstream.

No further historical-evidence architecture changes are required
without new empirical evidence.

Future work on sparse robust-z behavior, player-profile coverage, or
additional evidence sources should be treated as separate research
questions rather than extensions of Study 101F.