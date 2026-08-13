#binning.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BinningMode = Literal["custom", "quantile"]


@dataclass(frozen=True)
class BinningStrategy:
    mode: BinningMode
    bins: list[float] | None = None
    n_bins: int | None = None


@dataclass(frozen=True)
class BinDefinition:
    label: str
    left: float
    right: float


def format_bin_label(left: float, right: float) -> str:
    return f"[{left:.3f}, {right:.3f})"


def build_custom_bins(edges: list[float]) -> list[BinDefinition]:
    if len(edges) < 2:
        raise ValueError("Custom bins must contain at least two edges")

    if edges != sorted(edges):
        raise ValueError("Custom bin edges must be sorted ascending")

    return [
        BinDefinition(
            label=format_bin_label(left, right),
            left=float(left),
            right=float(right),
        )
        for left, right in zip(edges[:-1], edges[1:])
    ]


def build_quantile_bins(values: list[float], n_bins: int) -> list[BinDefinition]:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    if not values:
        raise ValueError("Cannot build quantile bins from empty values")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles).tolist()

    # Remove duplicate edges caused by repeated values.
    unique_edges = []
    for edge in edges:
        if not unique_edges or edge != unique_edges[-1]:
            unique_edges.append(float(edge))

    if len(unique_edges) < 2:
        raise ValueError("Quantile binning collapsed to fewer than two edges")

    return build_custom_bins(unique_edges)


def build_bins(
    values: list[float],
    strategy: BinningStrategy,
) -> list[BinDefinition]:
    if strategy.mode == "custom":
        if strategy.bins is None:
            raise ValueError("Custom binning requires bins")

        return build_custom_bins(strategy.bins)

    if strategy.mode == "quantile":
        if strategy.n_bins is None:
            raise ValueError("Quantile binning requires n_bins")

        return build_quantile_bins(values, strategy.n_bins)

    raise ValueError(f"Unsupported binning mode: {strategy.mode}")


def value_in_bin(value: float, bin_definition: BinDefinition) -> bool:
    return bin_definition.left <= value < bin_definition.right