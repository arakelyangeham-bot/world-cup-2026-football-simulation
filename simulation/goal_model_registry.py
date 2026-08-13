#goal_model_registry.py

from simulation.goal_models import PoissonGoalModel


GOAL_MODELS = {
    "goal_model_v1": PoissonGoalModel(
        name="goal_model_v1",
        home_features=[
            "home_poisson_attack",
            "away_poisson_defense",
            "fifa_points_diff",
        ],
        away_features=[
            "away_poisson_attack",
            "home_poisson_defense",
            "fifa_points_diff",
        ],
    ),
    "goal_model_v2": PoissonGoalModel(
        name="goal_model_v2",
        home_features=[
            "home_attack",
            "away_defense",
            "fifa_points_diff",
        ],
        away_features=[
            "away_attack",
            "home_defense",
            "fifa_points_diff",
        ],
    ),
    "goal_model_v3": PoissonGoalModel(
        name="goal_model_v3",
        home_features=[
            "home_poisson_attack",
            "away_poisson_defense",
            "fifa_points_diff",
        ],
        away_features=[
            "away_poisson_attack",
            "home_poisson_defense",
            "fifa_points_diff",
        ],
    ),
}


def get_goal_model(name: str) -> PoissonGoalModel:
    if name not in GOAL_MODELS:
        known = ", ".join(sorted(GOAL_MODELS))
        raise KeyError(
            f"Unknown goal model '{name}'. "
            f"Available: {known}"
        )

    return GOAL_MODELS[name]