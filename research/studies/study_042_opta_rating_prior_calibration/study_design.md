study_design.md

# Study 042 – Opta Rating Prior Calibration

## Status

In progress.

## Research question

How should raw Opta Power Rankings ratings be transformed into the
`rating_prior` input used by the Production Football Model?

## Motivation

The Production Football Model was originally trained using FIFA rating-point
differences as an external team-strength feature.

Club teams do not have FIFA points. Study 041 established a canonical global
club dataset based on Opta Power Rankings, but the raw Opta scale ranges from
approximately 0 to 100 and is not directly interchangeable with FIFA points.

Using raw Opta ratings without calibration would change the effective scale of
the model input and therefore alter the contribution of the fitted
rating-difference coefficient.

A calibrated transformation is required before the club prior can be treated
as production-ready.

## Inputs

- `research/data/processed/global_club_prior_dataset.csv`
- Existing national-team rating priors
- Historical club match results, when introduced during evaluation
- Production expected-goals model

## Study phases

### Phase 1 – Rating-scale audit

Describe the raw Opta rating distribution and investigate:

- central tendency,
- dispersion,
- percentiles,
- elite-level compression,
- rating differences at several rank separations,
- competition-specific coverage where metadata permit it.

No calibration method will be selected during this phase.

### Phase 2 – Candidate transformations

Construct candidate mappings from `opta_rating` to `rating_prior`.

Initial candidates may include:

1. Raw identity mapping
2. Simple linear scaling
3. Affine scaling
4. Distribution matching
5. Percentile-based mapping
6. Difference-scale calibration
7. Direct model retraining using Opta rating differences

### Phase 3 – Empirical evaluation

Compare candidate mappings using downstream football outcomes, including:

- historical match prediction,
- scoreline likelihood,
- calibration of win/draw/loss probabilities,
- simulated league-table realism,
- stability across competitions.

### Phase 4 – Production selection

Select and document the preferred mapping.

The final output must include:

- a reproducible transformation,
- fitted parameters,
- evaluation evidence,
- limitations,
- a versioned calibrated club-prior dataset.

## Separation of responsibilities

Study 041 is responsible for acquisition, canonicalization, and validation.

Study 042 is responsible for statistical calibration.

The raw `opta_rating` column must never be overwritten. Any calibrated value
must be written to the separate `rating_prior` column and accompanied by a
`rating_prior_method` label.

## Initial hypothesis

A direct identity mapping is unlikely to be compatible with the existing
Production Football Model because the model coefficient was fitted using the
much larger numerical scale of FIFA points.

A transformation based on rating differences, rather than merely matching
absolute ranges, is expected to provide a more defensible calibration.

## Completion criteria

Study 042 is complete when:

- the Opta scale has been fully described,
- candidate mappings have been implemented,
- mappings have been evaluated on football outcomes,
- one method has been selected or the evidence supports retraining,
- a calibrated dataset and research report have been produced.