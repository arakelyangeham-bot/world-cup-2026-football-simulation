#simulation_config.py

MATCH_ENGINE_MODE = "scoreline_first"
LAMBDA_MODEL = "calibrated"
LAMBDA_SCALE = 0.75
GOAL_SAMPLER = "dixon_coles_hierarchical"

GOAL_SAMPLER_CONFIG = {
    "tempo_cv": 0.60,
    "team_cv": 0.10,
    "rho": 0.30,
}

TEAM_REPOSITORY_SOURCE = "legacy"

# Previous Version 4 production sampler:
# GOAL_SAMPLER = "hierarchical_stochastic_lambda"
# GOAL_SAMPLER_CONFIG = {
#     "tempo_cv": 0.60,
#     "team_cv": 0.10,
# }