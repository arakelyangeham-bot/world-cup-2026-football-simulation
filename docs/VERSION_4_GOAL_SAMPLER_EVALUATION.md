VERSION_4_GOAL_SAMPLER_EVALUATION.md

Version 4 Goal Sampler Evaluation

Project: FIFA World Cup 2026 Data Science Project

Phase: Version 4 Goal Generation Research

Status: Completed

1. Objective

The objective of Version 4 was to improve the realism of simulated football scorelines while preserving the calibrated expected-goals (λ) model developed in Version 3.

The tournament framework, expected-goals model, and Monte Carlo engine remained unchanged throughout Version 4. Research focused exclusively on the goal-generation process.

Primary goals included:

improving scoreline realism
improving draw modelling
preserving expected goals
maintaining modular software architecture
benchmarking every proposed change against historical tournament data
2. Baseline

Version 3 production model:

Calibrated Poisson expected-goals model
Independent Poisson goal sampling

Historical dataset:

389 international tournament matches
27 tournaments
World Cup
UEFA EURO
Copa América
AFCON
AFC Asian Cup
3. Goal Samplers Investigated

Version 4 evaluated the following goal-generation approaches.

3.1 Negative Binomial

Objective:

Increase score variance relative to independent Poisson sampling.

Outcome:

increased variance
reduced overall realism
rejected
3.2 Shared Tempo

Objective:

Introduce match-level tempo affecting both teams.

Outcome:

improved realism
became foundation for later work
3.3 Fixed Mixture

Objective:

Model multiple latent match states.

Outcome:

promising
difficult to tune
superseded
3.4 Configurable Mixture

Objective:

Generalize mixture modelling.

Outcome:

modular implementation
benchmarked successfully
ultimately inferior to stochastic λ methods
3.5 Stochastic λ

Objective:

Treat λ as a random variable rather than fixed.

Outcome:

clear improvement
justified further development
3.6 Hierarchical Stochastic λ

Objective:

Separate randomness into:

match tempo
team-specific variation

Best parameters:

tempo_cv = 0.60

team_cv = 0.10

Outcome:

Best-performing Version 4 model.

3.7 Score-Level Draw Calibration

Objective:

Increase draw frequency by converting selected one-goal victories into draws.

Outcome:

Advantages

smoothly adjustable
improved draw rate

Disadvantages

reduced average goals
altered generated scorelines after sampling

Not selected.

3.8 Lambda Tempering

Objective:

Bring home and away λ values closer before sampling.

Outcome:

Minimal effect.

Experiment concluded.

3.9 Hierarchical Bivariate Poisson

Objective:

Introduce positive correlation through a shared Poisson component.

Outcome:

improved draw rate
preserved expected goals
increased total-goal variance

Did not outperform the hierarchical stochastic λ sampler under current benchmarks.

4. Benchmark Evolution

Version 4 introduced a progressively richer benchmarking framework.

Aggregate Statistics

Compared:

average goals
total variance
draw rate
clean sheet rate
5+ goal frequency
6+ goal frequency
Composite Error

Introduced:

absolute composite error
relative composite error

allowing automatic ranking of competing samplers.

Scoreline Distribution Benchmark

Added full scoreline comparison using Total Variation Distance.

Results:

Model	TV Distance
Historical	0.0000
Calibrated Poisson	0.2107
Hierarchical Stochastic λ	0.1851

Relative improvement:

Approximately 12.1% reduction in scoreline-distribution error.

5. Principal Findings

Major improvements:

more realistic 1–0 scorelines
more realistic 0–1 scorelines
improved modelling of moderate asymmetric victories
smoother overall scoreline distribution

Remaining weaknesses:

slight overproduction of 1–1 draws
draw composition still imperfect
total-goal variance sensitive to correlated-goal models
6. Software Engineering Outcomes

Version 4 also substantially improved project architecture.

New reusable components include:

modular goal samplers
hierarchical λ generation
generalized sampler benchmarks
composite benchmark metrics
scoreline-distribution benchmarking
automated CSV benchmark outputs

The benchmarking framework is now reusable for future sampler research.

7. Production Recommendation

Current production goal sampler:

Hierarchical Stochastic λ Sampler

Parameters:

tempo_cv = 0.60

team_cv = 0.10

Rationale:

best overall benchmark performance
strongest scoreline-distribution realism
maintains calibrated expected goals
modular implementation
computationally efficient
integrates directly into the tournament simulator
8. Future Research (Version 5)

Potential research directions include:

Dixon–Coles dependence adjustments
zero-inflated score models
dynamic match-state modelling
Bayesian parameter estimation
scoreline-specific calibration
tournament-context effects
penalty shootout calibration

Each proposed improvement should be evaluated using the established benchmarking framework before integration into production.

9. Version 4 Summary

Version 4 successfully transitioned the project from a simple independent Poisson score generator to a statistically richer hierarchical stochastic λ framework.

The new model produces substantially more realistic football score distributions while preserving computational efficiency and modular software design.

The benchmarking framework developed during Version 4 now provides a robust, evidence-driven foundation for future goal-model research and for evaluating all subsequent improvements to the FIFA World Cup 2026 simulation engine.