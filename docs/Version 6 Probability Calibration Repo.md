# Version 6 Probability Calibration Report

## Purpose

Version 6 investigated whether the production outcome probabilities could be improved through post-processing calibration while leaving the production goal engine and ML classifier unchanged.

Current production reference:

- Goal engine: Dixon–Coles hierarchical sampler
- rho = 0.30
- tempo_cv = 0.60
- team_cv = 0.10
- Evaluation baseline: v5.1_dixon_coles_hierarchical

## Calibration Dataset

A calibration dataset was created from the historical training dataset.

Rows:

- 389 historical matches

Fields included:

- home/away teams
- actual result
- production probabilities
- expected goals
- FIFA points difference
- team-strength features

Output:

```text
outputs/analysis/probability_calibration_dataset.csv

Baseline Calibration Metrics
Metric	Value
Multiclass Brier Score	0.575413
Multiclass Log Loss	0.989387
Mean ECE	0.119522
Experiment 1: Class Multiplier Calibration
Hypothesis

A simple home/draw/away probability multiplier could correct systematic calibration bias.

In-sample Result

The grid search found many Pareto-improving candidates on the full dataset.

A representative candidate:

home = 0.97
draw = 1.10
away = 1.05

This improved:

Brier score
Log loss
Mean ECE
Cross-validation Result

Out-of-sample cross-validation showed:

ECE improved
Log loss did not consistently improve
Brier score worsened on average

Decision:

Do not promote.

The method improves calibration error but does not robustly improve predictive performance.

Experiment 2: Pseudo-logit Temperature Scaling
Hypothesis

Temperature scaling applied to reconstructed pseudo-logits could improve probability calibration.

Pseudo-logits were derived from probabilities using:

log(p)
Cross-validation Result

Temperature scaling did not improve production performance.

Compared with identity:

Brier score worsened
Log loss worsened
Mean ECE improved only slightly

Decision:

Do not promote.

Pseudo-logit temperature scaling is not a production candidate.

Cross-validation Summary
Method	Brier	Log Loss	Mean ECE	Decision
Identity	0.575355	0.989259	0.127761	Keep production
Class Multiplier	0.578446	0.989861	0.122756	Do not promote
Temperature Scaling	0.577643	0.995948	0.125654	Do not promote
Conclusion

The current production probabilities remain the best option among tested methods.

Simple post-processing calibration methods revealed some systematic calibration bias, but neither tested method produced a robust out-of-sample improvement across the primary promotion metrics.

Recommendation

Do not modify production probability outputs yet.

Next research direction:

Expose raw classifier scores or logits where supported.
Re-test temperature scaling using true model scores rather than pseudo-logits.
Consider model-aware calibration methods only after preserving the existing production probability interface.
Status

Version 6 probability calibration research is complete for:

identity baseline
class multiplier calibration
pseudo-logit temperature scaling

No production promotion recommended.