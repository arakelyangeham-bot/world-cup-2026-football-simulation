#relationship.py

from __future__ import annotations

from dataclasses import dataclass

from research.football_observatory.binning import BinningStrategy
from research.football_observatory.observables import FootballObservable


@dataclass(frozen=True)
class FootballRelationship:
    name: str
    description: str
    independent_variable: str
    observable: FootballObservable
    binning: BinningStrategy