VERSION_5_DIXON_COLES_EVALUATION.md

# Version 5 Dixon–Coles Goal Sampler Evaluation

**Project:** FIFA World Cup 2026 Data Science Project  
**Phase:** Version 5 Goal Model Research  
**Status:** Production Candidate Validated  

---

## 1. Motivation

Version 4 promoted the hierarchical stochastic λ sampler as the production goal engine.

Subsequent scoreline-distribution diagnostics showed that the largest remaining error was concentrated in one scoreline:

| Scoreline | Historical | Version 4 Hierarchical |
|---|---:|---:|
| 0–0 | 6.17% | 11.77% |

The 0–0 scoreline accounted for roughly 15% of remaining score-distribution error.

---

## 2. Diagnostic Experiment

A temporary zero-zero deflation sampler was tested.

Best diagnostic result:

| Model | TV Distance |
|---|---:|
| Version 4 Hierarchical | 0.1851 |
| Zero-zero deflated 0.40 | 0.1579 |

This validated the hypothesis that excess 0–0 scorelines were a major source of model error.

The diagnostic sampler was not considered production-ready because it relied on post-sampling rerolls.

---

## 3. Dixon–Coles Candidate

A Dixon–Coles low-score adjustment was implemented on top of the hierarchical stochastic λ structure.

Selected candidate:

```text
sampler = dixon_coles_hierarchical_sampler_fast
tempo_cv = 0.60
team_cv = 0.10
rho = 0.30

The model adjusts low-score dependence while preserving the calibrated λ framework.

4. Scoreline Distribution Benchmark
Model	TV Distance
Poisson	0.2131
Version 4 Hierarchical	0.1850
Dixon–Coles ρ=0.30	0.1464

Improvement over Version 4:

≈ 21%

Improvement over original Poisson baseline:

≈ 31%
5. Aggregate Benchmark
Model	Relative Composite Error
Version 4 Hierarchical	0.915
Dixon–Coles ρ=0.30	0.678

The Dixon–Coles sampler also ranked best on aggregate score statistics.

Key aggregate behavior:

Metric	Historical	Version 4 Hierarchical	Dixon–Coles ρ=0.30
Avg goals	3.139	3.139	3.107
Total variance	7.285	7.519	7.008
Draw rate	0.175	0.270	0.217
Clean sheet rate	0.481	0.481	0.511
6. Runtime Benchmark
Sampler	Samples/sec	ms/sample
Hierarchical	195,851	0.0051
Dixon–Coles naive	5,359	0.1866
Dixon–Coles fast	30,074	0.0333

The vectorized Dixon–Coles implementation improved runtime by approximately 5.6× over the naive implementation.

It remains slower than the hierarchical sampler, but is acceptable for production-candidate testing.

7. End-to-End Monte Carlo Validation

The Dixon–Coles candidate was integrated into the production match engine and tested through the full tournament simulator.

Monte Carlo validation:

Metric	Value
Tournaments	1,000
Matches	104,000
Total goals	334,390
Avg goals per match	3.215
Extra-time rate	4.58%
Penalty shootout rate	1.08%

The simulator completed successfully with stable tournament-level behavior.

8. Production Recommendation

The Dixon–Coles hierarchical sampler with ρ=0.30 is recommended as the new production goal sampler.

Recommended config:

GOAL_SAMPLER = "dixon_coles_hierarchical"

GOAL_SAMPLER_CONFIG = {
    "tempo_cv": 0.60,
    "team_cv": 0.10,
    "rho": 0.30,
}
9. Rationale

The candidate has passed:

aggregate statistics benchmark
relative composite benchmark
scoreline-distribution benchmark
runtime benchmark
full Monte Carlo integration test

It improves the scoreline-distribution benchmark by roughly 21% over Version 4 production while also improving aggregate benchmark performance.

10. Future Work

Future improvements should focus on:

end-to-end tournament calibration
confidence intervals for Monte Carlo probabilities
larger-scale runtime testing
automated production gate benchmarking
possible optimization of Dixon–Coles sampling
11. Summary

Version 5 successfully identified and addressed the largest remaining scoreline error in the Version 4 model: overproduction of 0–0 results.

The Dixon–Coles low-score correction provides a statistically principled improvement over the Version 4 hierarchical stochastic λ sampler and is recommended for production promotion.

```text
Promote Dixon-Coles hierarchical sampler to production