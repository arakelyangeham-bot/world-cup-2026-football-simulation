experiment_031_report.md

# Experiment 031 — League vs Knockout Competitions

## Program

Competition Research

## Status

Completed

---

# Research Question

Does a league competition identify the strongest team more reliably than a knockout competition?

---

# Motivation

Football competitions reward different qualities.

League competitions reward long-term consistency by allowing every team to play multiple matches against the field.

Knockout competitions reward survival under uncertainty, where one poor performance may eliminate even the strongest team.

Although this intuition is common in football discussion, the project seeks to evaluate it computationally under controlled experimental conditions.

Experiment 031 is the first Version 3 research program designed to compare competition formats while holding the underlying football model constant.

---

# Program Structure

Experiment 031 consisted of three sequential studies.

## Experiment 031A

Research Framework Validation

Purpose:

Validate the newly developed Version 3 research framework.

Validated components:

- Experiment
- ExperimentCondition
- ExperimentRunner
- Metric Library
- ExperimentReport

No football conclusions were intended.

---

## Experiment 031B

Synthetic League vs Knockout

Purpose:

Validate the research methodology using a simplified synthetic football model.

Football model:

- synthetic team strength ladder
- synthetic match model
- deterministic experimental configuration

Result:

League competitions identified the strongest team more frequently than knockout competitions.

---

## Experiment 031C

Production League vs Knockout

Purpose:

Replicate Experiment 031B using the project's production football model.

Production Football Model v1 consisted of:

- dimension-specific Player Intelligence repository
- production scoreline-first match engine
- expected-goals model
- Dixon-Coles hierarchical goal sampler
- Competition Framework
- Research Framework

Only the football model changed.

The research methodology remained identical.

---

# Experimental Design

## Fixed Variables

- Eight-team field
- Same participating teams
- Same experimental methodology
- Same metrics
- Same simulation count
- Same random seed policy

## Independent Variable

Competition format.

Two conditions were compared:

- Single round-robin league
- Seeded knockout tournament

## Dependent Variables

- Average champion strength
- Strongest-team championship rate
- Champion variance
- Upset rate

---

# Results

## Experiment 031B

| Metric | League | Knockout |
|---------|--------|-----------|
| Average champion strength | 89.995 | 85.585 |
| Strongest-team championship rate | 29.4% | 17.7% |
| Champion variance | 8 | 8 |
| Upset rate | 45.1% | 44.9% |

---

## Experiment 031C

| Metric | League | Knockout |
|---------|--------|-----------|
| Average normalized champion strength | 0.8855 | 0.8592 |
| Strongest-team championship rate | 45.6% | 35.6% |
| Champion variance | 8 | 8 |
| Upset rate | 31.5% | 37.4% |

---

# Interpretation

The central conclusion of Experiment 031B remained valid after replacing the synthetic football model with the production football model.

Across both implementations:

- league competitions produced stronger champions on average,
- league competitions crowned the strongest team more frequently than knockout competitions.

This suggests that the qualitative relationship between competition format and champion quality is robust across substantially different football models.

Experiment 031C also demonstrated lower upset rates than the synthetic experiment.

This indicates that the production football model produces greater separation between stronger and weaker teams, likely due to improvements introduced by Player Intelligence, expected-goals calibration, and scoreline-first match generation.

Although the quantitative values changed, the qualitative conclusion remained unchanged.

---

# What Surprised Us?

The most unexpected observation was that upset rates changed considerably between the synthetic and production football models while the primary research conclusion remained stable.

This suggests that competition format and football model influence tournament outcomes in different ways.

Competition structure appears to determine how individual match outcomes accumulate into tournament champions, while football model fidelity primarily influences the probability of individual upsets.

This distinction was not an original objective of the experiment but emerged naturally from the comparison.

---

# Limitations

Experiment 031 intentionally simplified several aspects of football.

These include:

- synthetic eight-team field
- simplified competition sizes
- no home advantage
- no injuries
- no lineup uncertainty
- no squad rotation
- no dynamic player availability

The purpose of the experiment was controlled comparison rather than complete realism.

---

# Conclusions

Experiment 031 established the first complete computational football research workflow within the project.

It demonstrated that:

- the Version 3 Research Framework functions correctly,
- research questions can be expressed through reusable experiments,
- competition formats can be compared under controlled assumptions,
- production football models can be incorporated without changing the experimental methodology.

Most importantly, the experiment produced the project's first replicated football research result:

> League competitions identified the strongest team more reliably than knockout competitions under both synthetic and production football models.

---

# Future Work

Experiment 032 will investigate which components of the production football model most strongly influence competitive discrimination.

Possible factors include:

- Player Intelligence repository
- scoreline-first match engine
- goal sampling methodology
- expected-goals calibration

Future Competition Research experiments will investigate:

- seeding sensitivity,
- tournament expansion,
- group-stage value,
- bracket fairness,
- competition robustness.

Experiment 031 establishes the baseline methodology for these future studies.

---

# Significance

Experiment 031 marks the transition of the project from simulation software toward computational football research.

Previous versions focused on constructing infrastructure.

Experiment 031 demonstrates that the infrastructure can now be used to generate reproducible evidence about football competitions.

It represents the first research program completed using the Version 3 research framework.