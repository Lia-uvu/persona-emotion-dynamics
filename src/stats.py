"""Trajectory slope estimation and exact block-level permutation tests."""

from __future__ import annotations

from itertools import product
from statistics import mean, stdev
from typing import Mapping, Sequence


def ols_slope(values: Sequence[float]) -> float:
    """OLS slope across equally spaced one-indexed rounds."""
    if len(values) < 2:
        raise ValueError("trajectory slope requires at least two rounds")
    x_mean = (len(values) + 1) / 2
    y_mean = mean(values)
    denominator = sum(
        (round_number - x_mean) ** 2
        for round_number in range(1, len(values) + 1)
    )
    return sum(
        (round_number - x_mean) * (value - y_mean)
        for round_number, value in enumerate(values, start=1)
    ) / denominator


def factorial_contrasts(quadrants: Mapping[str, float]) -> dict[str, float]:
    """Compute grand mean, E, N, and E x N contrasts from 2x2 quadrant slopes."""
    hh = quadrants["high_e_high_n"]
    hl = quadrants["high_e_low_n"]
    lh = quadrants["low_e_high_n"]
    ll = quadrants["low_e_low_n"]
    return {
        "overall": mean((hh, hl, lh, ll)),
        "extraversion": (hh + hl - lh - ll) / 2,
        "neuroticism": (hh + lh - hl - ll) / 2,
        "interaction": hh - hl - lh + ll,
    }


def descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    deviation = stdev(values) if len(values) > 1 else None
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "sample_stdev": deviation,
        "standard_error": deviation / (len(values) ** 0.5) if deviation is not None else None,
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
    }


def exact_sign_flip_p(values: Sequence[float]) -> float | None:
    """Two-sided exact sign-flip p-value across paired blocks."""
    if not values:
        return None
    observed = abs(mean(values))
    extreme = 0
    for signs in product((-1, 1), repeat=len(values)):
        permuted = abs(
            mean(value * sign for value, sign in zip(values, signs, strict=True))
        )
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / (2 ** len(values))


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Sequentially rejective Holm step-down adjustment within a test family."""
    available = sorted(
        ((name, value) for name, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    family_size = len(available)
    for index, (name, value) in enumerate(available):
        running = max(running, min(1.0, (family_size - index) * value))
        adjusted[name] = running
    return adjusted
