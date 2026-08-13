Version 5 Research Note
Zero-Score Diagnostic Experiment

Project: FIFA World Cup 2026 Data Science Project

Phase: Version 5 Research

Status: Diagnostic Experiment

1. Motivation

Following completion of the Version 4 evaluation framework, scoreline-distribution analysis was performed on the production Hierarchical Stochastic λ sampler.

The analysis revealed that a single scoreline dominated the remaining model error.

Scoreline	Historical	Hierarchical
0–0	6.17%	11.77%

The 0–0 scoreline alone accounted for approximately 15% of the remaining score-distribution error, making it the largest individual discrepancy between simulated and historical tournament football.

This observation motivated a targeted diagnostic experiment.

2. Research Question

Rather than attempting to improve football score realism in general, Version 5 focused on a much narrower question:

Would reducing excess 0–0 scorelines materially improve the overall scoreline distribution?

3. Experimental Design

A temporary experimental sampler was implemented.

Algorithm:

Generate a scoreline using the production Hierarchical Stochastic λ sampler.
If the generated scoreline is 0–0, then with probability p:
discard the result;
generate one replacement scoreline;
accept the replacement.

The experiment intentionally used a simple heuristic rather than a statistically principled model.

Its purpose was diagnostic rather than production deployment.

4. Parameter Sweep

The following resampling probabilities were evaluated:

Parameter
0.00
0.10
0.20
0.30
0.40

Each configuration was benchmarked using the Version 4 scoreline-distribution benchmark.

5. Results
Model	Total Variation Distance
Historical	0.000000
Hierarchical Stochastic λ	0.185139
Zero-Score Deflation (0.10)	0.178596
Zero-Score Deflation (0.20)	0.171422
Zero-Score Deflation (0.30)	0.165111
Zero-Score Deflation (0.40)	0.157949

Relative improvement over the production hierarchical sampler:

≈ 14.7%

Relative improvement over the original calibrated Poisson sampler:

≈ 25%

The benchmark improvement was smooth and monotonic across the tested parameter range.

6. Interpretation

The experiment strongly supports the hypothesis that excessive 0–0 scorelines are a major contributor to the remaining score-distribution error.

However, the experiment does not justify deploying the heuristic itself.

The implemented sampler modifies generated outcomes after sampling and therefore does not represent an underlying probabilistic model of football scores.

Instead, the experiment demonstrates that:

Reducing excess scoreless draws is a promising direction for future model development.

7. Engineering Outcome

The experiment successfully validated:

the Version 4 benchmarking framework,
the scoreline-distribution benchmark,
the usefulness of scoreline-specific diagnostics.

It also demonstrated the value of targeted experiments before developing more sophisticated statistical models.

8. Production Decision

The experimental sampler was not promoted to production.

The production goal engine remains:

Calibrated λ model
Hierarchical Stochastic λ sampler
tempo_cv = 0.60
team_cv = 0.10

The diagnostic sampler is retained solely as research evidence.

9. Future Work

The observed improvement suggests that future research should pursue a statistically principled mechanism capable of reducing excess 0–0 scorelines without post-processing generated outcomes.

Potential directions include:

Dixon–Coles dependence adjustment
Low-score correction models
Zero-deflated probabilistic models
Learned λ-dependent low-score calibration

Future candidates should be evaluated against the established benchmark suite and compared directly with the current production baseline.

10. Summary

This experiment marks the beginning of Version 5 research.

Although the heuristic sampler is not suitable for production use, it provided strong empirical evidence that the largest remaining weakness of the current production goal model lies in the generation of scoreless draws.

The experiment therefore establishes a focused, evidence-based direction for subsequent research into low-score dependence modeling.