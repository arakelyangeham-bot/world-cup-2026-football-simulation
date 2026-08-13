script_inventory.md

# Script Inventory and Classification

## Purpose

This document classifies project scripts by their current role.

Categories:

- **KEEP** — active production or reusable workflow
- **RESEARCH** — useful experiment, not production
- **ARCHIVE** — superseded but historically valuable
- **REVIEW** — unclear, inspect before deciding
- **DELETE** — obsolete or misleading

No files should be moved or deleted during this pass. This is documentation only.

---

## High-Level Assessment

The `scripts/` directory currently contains a mixture of:

- benchmark scripts
- audit scripts
- research experiments
- data collection scripts
- Sofascore-era scripts
- World Cup 2026 construction utilities
- historical validation scripts

The goal is to identify what belongs to the current platform architecture and what should eventually be archived or reorganized.

---

## Active Subdirectories

### scripts/analysis/

Classification: **KEEP**

Reason:

Contains active analysis scripts used for probability calibration, reliability tables, and benchmark summaries.

### scripts/benchmarks/

Classification: **KEEP**

Reason:

Contains the active regression framework and comparison modules.

### scripts/research/

Classification: **KEEP / RESEARCH**

Reason:

Contains active Version 6 and Version 7 research workflows.

### scripts/utilities/

Classification: **KEEP**

Reason:

Contains reusable baseline and promotion utilities.

---

## Top-Level Script Classification

| Script | Classification | Notes |
|---|---|---|
| analyze_lambda_contributions.py | REVIEW | Older lambda-analysis utility; inspect before archiving. |
| analyze_match_engine.py | REVIEW | Historical match-engine analysis; likely superseded by newer benchmark scripts. |
| analyze_scoreline_errors.py | REVIEW | Useful diagnostic; may belong under analysis or archive. |
| analyze_team_strengths.py | REVIEW | Team-strength diagnostic; may remain useful for future football-intelligence phase. |
| audit_competition_registry.py | KEEP | Validates competition registry. |
| audit_dataset_growth.py | KEEP | Dataset integrity / growth audit. |
| audit_fast_predictor_equivalence.py | KEEP | Validates inference optimization equivalence. |
| audit_feature_correlation_matrix.py | RESEARCH | Useful feature analysis. |
| audit_feature_goal_relationships.py | RESEARCH | Useful feature/goal relationship analysis. |
| audit_feature_vector_builder.py | KEEP | Validates production feature vector pipeline. |
| audit_full_knockout.py | KEEP | Tournament framework validation. |
| audit_goal_model_interface.py | KEEP | Validates goal model interface. |
| audit_goal_variance.py | RESEARCH | Goal-model research diagnostic. |
| audit_historical_calibration_dataset.py | KEEP | Validates calibration dataset. |
| audit_historical_competitions.py | KEEP | Dataset/competition audit. |
| audit_historical_dataset_coverage.py | KEEP | Dataset coverage audit. |
| audit_historical_match_catalog.py | KEEP | Historical match catalog audit. |
| audit_knockout_mapping.py | KEEP | Validates knockout mapping. |
| audit_knockout_match_simulation.py | KEEP | Validates knockout match simulation. |
| audit_lambda_override.py | REVIEW | Inspect whether still needed after current lambda model architecture. |
| audit_match_engine_modes.py | KEEP | Validates match engine configuration modes. |
| audit_ml_predictor_runtime.py | KEEP | Runtime audit for ML predictor. |
| audit_model_feature_order.py | KEEP | Validates model feature ordering. |
| audit_monte_carlo_outputs.py | KEEP | Validates Monte Carlo output structure. |
| audit_negative_binomial_support.py | ARCHIVE | Superseded by Version 5 production sampler unless reused later. |
| audit_outcome_model_historical_performance.py | KEEP | Authoritative historical outcome-model audit wrapper. |
| audit_outcome_probability_calibration.py | REVIEW | Likely superseded by scripts/analysis calibration benchmark. |
| audit_poisson_calibration.py | ARCHIVE | Superseded by calibrated goal model and Version 5 benchmarks. |
| audit_poisson_features.py | KEEP | Validates Poisson-derived features. |
| audit_raw_match_inventory.py | KEEP | Dataset integrity audit. |
| audit_round_of_16_builder.py | KEEP | Tournament construction audit. |
| audit_round_of_32_builder.py | KEEP | Tournament construction audit. |
| audit_sofascore_seasons.py | REVIEW | Belongs to future data_collection/sofascore subsystem. |
| audit_team_name_matching.py | KEEP | Validates name normalization/matching. |
| audit_team_repository.py | KEEP | Validates team repository. |
| audit_team_strength_distribution.py | KEEP | Team-strength diagnostic. |
| audit_wc2026_structure.py | KEEP | Validates World Cup 2026 structure. |

---

## Benchmark Scripts

| Script | Classification | Notes |
|---|---|---|
| benchmark_configurable_mixture_sampler.py | ARCHIVE | Version 4 research benchmark; superseded by Dixon-Coles production sampler. |
| benchmark_draw_calibrated_sampler.py | ARCHIVE | Version 5 research artifact; keep for history. |
| benchmark_goal_sampler_runtime.py | KEEP | Useful runtime benchmark for future sampler changes. |
| benchmark_goal_variance.py | RESEARCH | Useful diagnostic for goal model variance. |
| benchmark_hierarchical_goal_samplers.py | ARCHIVE | Version 4 sampler research. |
| benchmark_hierarchical_stochastic_lambda.py | ARCHIVE | Version 4 production predecessor. |
| benchmark_hybrid_goal_sampler.py | ARCHIVE | Experimental sampler benchmark. |
| benchmark_match_models.py | REVIEW | Inspect whether still useful for outcome-model benchmarking. |
| benchmark_negative_binomial_sampler.py | ARCHIVE | Superseded by Version 5 sampler. |
| benchmark_scoreline_distribution.py | KEEP | Active scoreline realism benchmark. |
| benchmark_score_models.py | REVIEW | May overlap with scoreline-distribution benchmark. |
| benchmark_score_model_fitness.py | REVIEW | Inspect before archiving. |
| benchmark_shared_tempo_sampler.py | ARCHIVE | Version 4 research artifact. |
| benchmark_stochastic_lambda_sampler.py | ARCHIVE | Version 4 research artifact. |

---

## Dataset and Model-Building Scripts

| Script | Classification | Notes |
|---|---|---|
| build_historical_training_dataset.py | KEEP | Builds canonical historical training dataset. |
| build_match_feature_table.py | KEEP | Builds match-level feature table; important for ML pipeline. |
| fit_poisson_goal_model.py | ARCHIVE | Superseded by current calibrated lambda / goal-model pipeline unless still used. |
| cross_validate_goal_models.py | RESEARCH | Useful for goal-model research and future validation. |
| evaluate_poisson_goal_buckets.py | ARCHIVE | Older Poisson diagnostic. |
| evaluate_poisson_goal_model.py | ARCHIVE | Older Poisson diagnostic. |
| compare_poisson_feature_sets.py | REVIEW | May be useful for future goal-feature research. |
| inspect_fifa_points_coverage.py | KEEP | Useful audit for FIFA points coverage. |
| inspect_national_team_priors.py | KEEP | Useful audit for national-team priors. |
| inspect_team_strength_loaders.py | KEEP | Useful team-strength loader audit. |

---

## Simulation and Tournament Scripts

| Script | Classification | Notes |
|---|---|---|
| bracket_engine_smoke_test.py | KEEP | Smoke test for bracket engine behavior. |
| compare_simulation_to_history.py | REVIEW | Historical comparison utility; inspect for overlap with benchmarks. |
| generate_third_place_assignments.py | ARCHIVE | One-time construction utility unless still needed. |
| historical_match_engine_validation.py | KEEP | Important validation against historical matches. |
| historical_wc_analysis.py | REVIEW | Historical analysis utility; inspect before archiving. |
| match_engine.py | REVIEW | Check whether superseded by simulation/match_engine_adapter.py and production engine. |
| monte_carlo_driver.py | KEEP | Active Monte Carlo driver. |
| plot_sofascore_wc_2026.py | REVIEW | Visualization utility; likely data/reporting, not core. |
| profile_coefficient_loading.py | ARCHIVE | Performance diagnostic; probably historical. |
| profile_single_tournament.py | KEEP | Useful runtime profiling utility. |
| results_summary.py | REVIEW | Inspect current usage. |
| simulation_scoreline_distribution.py | REVIEW | May overlap with active scoreline benchmarks. |
| simulation_utils.py | REVIEW | Inspect whether functions are still imported. |
| validate_group_stage.py | KEEP | Group-stage validation. |
| wc2026_bracket.py | REVIEW | Older bracket utility; compare against current production tournament framework. |
| wc2026_data.py | KEEP | World Cup 2026 data definitions. |
| wc2026_group_stage.py | KEEP | World Cup 2026 group-stage builder/simulator. |
| wc2026_knockout_mapping.py | KEEP | Knockout mapping logic. |
| wc2026_knockout_stage.py | KEEP | Knockout-stage builder/simulator. |
| wc2026_seed_assignment.py | KEEP | Seed assignment logic. |
| wc2026_third_place_assignments.py | KEEP | Third-place assignment rules. |
| wc2026_tournament_simulator.py | KEEP | World Cup 2026 tournament simulator. |

---

## Sofascore and Data Collection Scripts

| Script | Classification | Notes |
|---|---|---|
| discover_player_competitions.py | REVIEW | Future data-collection subsystem candidate. |
| discover_player_ids.py | REVIEW | Future data-collection subsystem candidate. |
| fetch_sofascore_international_season_ids.py | REVIEW | Belongs under future data_collection/sofascore. |
| fetch_sofascore_league_season_ids.py | REVIEW | Belongs under future data_collection/sofascore. |
| fifa_rankings_scraper.py | KEEP | Useful data collection for FIFA rankings. |
| run_sofascore_pipeline.py | REVIEW | Important historical pipeline; needs subsystem classification. |
| sofascore_backtest.py | ARCHIVE | Sofascore-era model research; keep historically. |
| sofascore_bracket_audit.py | ARCHIVE | Superseded by rebuilt tournament framework. |
| sofascore_competition_scraper.py | REVIEW | Future data_collection/sofascore. |
| sofascore_config_loader.py | REVIEW | Future data_collection/sofascore. |
| sofascore_correlation_analysis.py | ARCHIVE | Historical Sofascore analysis. |
| sofascore_country_position_analysis.py | ARCHIVE | Historical Sofascore analysis. |
| sofascore_eda.py | ARCHIVE | Exploratory analysis. |
| sofascore_feature_engineering.py | REVIEW | May contain useful feature ideas for future player/team modeling. |
| sofascore_match_prediction.py | ARCHIVE | Superseded by current inference pipeline. |
| sofascore_match_scraper.py | REVIEW | Future data_collection/sofascore. |
| sofascore_merge_roster.py | REVIEW | Future player/team modeling candidate. |
| sofascore_model_optimizer.py | ARCHIVE | Historical optimizer; likely superseded. |
| sofascore_team_aggregator.py | REVIEW | Important candidate for future player-derived team-strength modeling. |
| sofascore_test_scraper.py | ARCHIVE | Scraper test utility. |
| sofascore_tournament_simulator.py | ARCHIVE | Superseded by rebuilt production tournament simulator. |
| sofascore_utils.py | REVIEW | Future data_collection/sofascore utility module. |
| sofascore_wc_scraper.py | REVIEW | Future data_collection/sofascore. |
| team_strength_loader.py | KEEP | Active or reusable team-strength loading utility. |

---

## Current Research Scripts

| Script | Classification | Notes |
|---|---|---|
| scripts/research/audit_decision_policy.py | RESEARCH | Version 7 decision-policy research. |
| scripts/research/audit_feature_importance.py | RESEARCH | Version 7 feature-importance research. |
| scripts/research/audit_prediction_pipeline.py | RESEARCH | Prediction-path validation; useful historical debugging artifact. |
| scripts/research/audit_production_classifier.py | RESEARCH | Production classifier introspection. |
| scripts/research/benchmark_outcome_models.py | REVIEW | Placeholder or future model benchmark; inspect before use. |
| scripts/research/calibrate_outcome_probabilities.py | RESEARCH | Version 6 calibration research. |
| scripts/research/compare_production_and_retrained_model.py | RESEARCH | Reproduction-debugging artifact. |
| scripts/research/cross_validate_probability_calibration.py | RESEARCH | Active calibration research framework. |
| scripts/research/evaluate_trained_outcome_model.py | REVIEW | Likely superseded by shared.outcome_evaluation; inspect before archiving. |
| scripts/research/training_config.py | KEEP | Reusable research training configuration. |
| scripts/research/train_outcome_model.py | KEEP | Reproducible research training wrapper. |

---

## Immediate Recommendations

1. Do not delete anything yet.
2. Treat `simulation/`, `inference/`, and `shared/` as stable core packages.
3. Treat `scripts/research/`, `scripts/benchmarks/`, and `scripts/utilities/` as active.
4. Gradually move old top-level scripts into clearer subdirectories only after verifying imports.
5. Create a future `data_collection/sofascore/` subsystem before expanding player-stat work.
6. Before Version 7 feature ablation, finish consolidating outcome-model evaluation around `shared.outcome_evaluation`.

## Next Review Target

The highest-priority `REVIEW` files are:

- `scripts/research/evaluate_trained_outcome_model.py`
- `scripts/benchmark_match_models.py`
- `scripts/benchmark_score_models.py`
- `scripts/match_engine.py`
- `scripts/simulation_utils.py`
- Sofascore feature/team aggregation scripts

These should be inspected before major reorganization.