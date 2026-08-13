#football_realization.py

from __future__ import annotations


def football_realization_adjustment(
    lambda_home: float,
    lambda_away: float,
    context: dict | None = None,
) -> tuple[float, float]:
    """
    Research scoreline-realization hook.

    Version 0 intentionally returns production expected-goal values unchanged.

    Future research prototypes may use this seam to apply empirically motivated
    football-process adjustments before scoreline sampling.
    """

    return lambda_home, lambda_away