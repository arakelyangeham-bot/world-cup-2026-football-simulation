#uncertainty.py

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BinomialUncertainty:
    count: int
    total: int
    rate: float
    standard_error: float
    ci_lower: float
    ci_upper: float


def wilson_interval(
    count: int,
    total: int,
    confidence: float = 0.95,
) -> BinomialUncertainty:
    """
    Wilson score interval for a binomial proportion.

    Uses z = 1.96 for 95% confidence.
    """

    if total < 0:
        raise ValueError("total must be non-negative")

    if count < 0:
        raise ValueError("count must be non-negative")

    if count > total:
        raise ValueError("count cannot exceed total")

    if total == 0:
        return BinomialUncertainty(
            count=count,
            total=total,
            rate=float("nan"),
            standard_error=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
        )

    if confidence != 0.95:
        raise ValueError("Only confidence=0.95 is currently supported")

    z = 1.96
    p = count / total

    standard_error = math.sqrt((p * (1.0 - p)) / total)

    denominator = 1.0 + (z * z / total)
    center = p + (z * z) / (2.0 * total)
    margin = z * math.sqrt(
        (p * (1.0 - p) / total)
        + (z * z) / (4.0 * total * total)
    )

    ci_lower = (center - margin) / denominator
    ci_upper = (center + margin) / denominator

    return BinomialUncertainty(
        count=count,
        total=total,
        rate=p,
        standard_error=standard_error,
        ci_lower=max(0.0, ci_lower),
        ci_upper=min(1.0, ci_upper),
    )