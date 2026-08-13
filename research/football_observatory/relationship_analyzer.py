#relationship_analyzer.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.football_observatory.binning import (
    BinDefinition,
    build_bins,
    value_in_bin,
)
from research.football_observatory.observatory_schema import MatchObservation
from research.football_observatory.relationship import FootballRelationship
from research.football_observatory.uncertainty import wilson_interval


@dataclass(frozen=True)
class RelationshipBinResult:
    relationship: str
    bin_label: str
    bin_min: float
    bin_max: float
    matches: int
    observable_count: int
    observable_rate: float
    standard_error: float
    ci_lower: float
    ci_upper: float


def get_independent_value(
    observation: MatchObservation,
    variable_name: str,
) -> float:
    if hasattr(observation.prematch, variable_name):
        return float(getattr(observation.prematch, variable_name))

    if (
        observation.derived_prematch is not None
        and variable_name in observation.derived_prematch
    ):
        return float(observation.derived_prematch[variable_name])

    raise ValueError(
        f"Unknown prematch variable for relationship analysis: "
        f"{variable_name}"
    )


def observations_for_bin(
    observations: list[MatchObservation],
    variable_name: str,
    bin_definition: BinDefinition,
) -> list[MatchObservation]:
    return [
        observation
        for observation in observations
        if value_in_bin(
            get_independent_value(observation, variable_name),
            bin_definition,
        )
    ]


def analyze_relationship(
    observations: list[MatchObservation],
    relationship: FootballRelationship,
) -> pd.DataFrame:
    independent_values = [
        get_independent_value(
            observation,
            relationship.independent_variable,
        )
        for observation in observations
    ]

    bins = build_bins(
        values=independent_values,
        strategy=relationship.binning,
    )

    rows: list[RelationshipBinResult] = []

    for bin_definition in bins:
        bin_observations = observations_for_bin(
            observations=observations,
            variable_name=relationship.independent_variable,
            bin_definition=bin_definition,
        )

        matches = len(bin_observations)

        if matches == 0:
            observable_count = 0
        else:
            observable_values = [
                relationship.observable.evaluate(observation)
                for observation in bin_observations
            ]

            observable_count = sum(observable_values)

        uncertainty = wilson_interval(
            count=observable_count,
            total=matches,
        )

        rows.append(
            RelationshipBinResult(
                relationship=relationship.name,
                bin_label=bin_definition.label,
                bin_min=bin_definition.left,
                bin_max=bin_definition.right,
                matches=matches,
                observable_count=observable_count,
                observable_rate=uncertainty.rate,
                standard_error=uncertainty.standard_error,
                ci_lower=uncertainty.ci_lower,
                ci_upper=uncertainty.ci_upper,
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])