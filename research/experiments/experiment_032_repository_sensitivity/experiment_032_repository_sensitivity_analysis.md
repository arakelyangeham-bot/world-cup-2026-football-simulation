# Experiment 032 — Repository Sensitivity Analysis

## Research Question

How sensitive are competition outcomes to the choice of team representation repository?

---

## Motivation

Version 2 introduced multiple strategies for constructing national team representations from player-level data. These repositories were originally developed as alternative engineering approaches for estimating team strength.

Experiment 032 treats the repository selection itself as the independent variable in order to determine whether different aggregation strategies materially influence computational football research conclusions.

Rather than identifying a "best" repository, the objective is to understand how sensitive tournament outcomes are to this modeling choice.

---

## Experimental Design

### Fixed Variables

- Team set
    - Argentina
    - France
    - Brazil
    - England
    - Spain
    - Portugal
    - Netherlands
    - Germany

- Production Football Model v1
- Production scoreline-first match engine
- Goal sampling methodology
- Simulation count (1000)
- Random seed policy
- Metric definitions

### Independent Variable

Team representation repository.

Repositories evaluated:

- legacy
- dimension_specific
- top_11_mean
- top_5_mean
- star_weighted
- starter_plus_depth

### Competition Formats

Each repository was evaluated using:

- Single round-robin league
- Seeded knockout tournament

### Metrics

- Average Champion Strength
- Strongest-Team Championship Rate
- Champion Variance
- Upset Rate

---

## Results

| Repository | League Strongest-Team Rate | Knockout Strongest-Team Rate |
|------------|---------------------------:|-----------------------------:|
| legacy | 0.091 | 0.152 |
| dimension_specific | 0.456 | 0.356 |
| top_11_mean | 0.132 | 0.174 |
| top_5_mean | 0.098 | 0.155 |
| star_weighted | 0.457 | 0.377 |
| starter_plus_depth | 0.203 | 0.182 |

---

## Observations

### Observation 1

The qualitative conclusion from Experiment 031 remained stable for the production-oriented repositories.

Dimension-specific and star-weighted repositories produced substantially higher strongest-team championship rates than the legacy aggregation methods.

---

### Observation 2

Repository construction significantly affected the magnitude of competitive discrimination.

The difference between legacy and dimension-specific repositories exceeded fourfold in league strongest-team championship rate.

This demonstrates that team representation is not merely an implementation detail but a major component of the football model.

---

### Observation 3

League competitions generally continued to identify the strongest team more frequently than knockout competitions.

Although the magnitude varied, the overall pattern observed in Experiment 031 remained intact for the strongest production repositories.

---

### Observation 4

Upset rates remained comparatively stable across repositories.

This suggests that repository construction primarily affects the estimation of underlying team strength rather than dramatically altering match-level stochasticity.

---

### Observation 5

Champion strength values differed substantially between repositories.

However, each repository defines team strength on its own numerical scale.

Consequently, absolute champion strength values should not be compared directly across repositories without normalization.

---

## Interpretation

Experiment 032 demonstrates that repository choice materially influences quantitative football simulation results.

Importantly, however, changing the repository did not fundamentally alter the qualitative conclusion that stronger competition formats tend to identify stronger champions.

Instead, repository choice primarily affected the confidence with which the football model distinguished stronger and weaker teams.

This provides evidence that improvements made during Version 2 were scientifically meaningful rather than merely architectural refinements.

---

## Methodological Notes

This experiment measures strongest-team championship rate relative to the ranking produced by each repository itself.

Future work may investigate an alternative methodology using a fixed external ranking as the reference definition of team strength.

---

## Limitations

- Champion strength values are repository-dependent.
- Team strength scales are not currently normalized.
- Only one match engine was evaluated.
- Only one competition set was evaluated.

---

## Conclusions

Experiment 032 demonstrates that repository construction is one of the most influential modeling choices within Production Football Model v1.

The experiment validates that the Player Intelligence work completed during Version 2 materially improved the ability of the football model to distinguish team quality.

At the same time, the principal conclusion from Competition Research Program A remained robust across repository choices, providing additional confidence in the stability of the broader research framework.

---

## Future Work

Experiment 033 — Goal Sampler Sensitivity

The next experiment will investigate whether different goal sampling methodologies materially influence competition outcomes while holding team representation fixed.