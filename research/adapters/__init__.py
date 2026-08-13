#__init__.py

from research.adapters.competition_adapter import CompetitionAdapter
from research.adapters.football_model_adapter import (
    FootballModel,
    FootballModelAdapter,
)

__all__ = [
    "CompetitionAdapter",
    "FootballModel",
    "FootballModelAdapter",
]