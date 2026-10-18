from __future__ import annotations

import math
from typing import Any

from .schemas import GateDecision


def finite_interval(interval: tuple[float, float]) -> bool:
    lower, upper = interval
    return math.isfinite(lower) and math.isfinite(upper) and lower <= upper


def gate_regression(h: int, interval: tuple[float, float], point: float, delta: float) -> GateDecision:
    # Publication decisions remain Python-controlled.
    if h:
        return GateDecision(False, 1, "execution_check_failed", interval)
    if not finite_interval(interval):
        return GateDecision(False, 0, "invalid_interval", interval)
    lower, upper = interval
    allowed = (point - delta, point + delta)
    release = allowed[0] <= lower and upper <= allowed[1]
    reason = "conformal_interval_within_delta" if release else "interval_exceeds_delta"
    return GateDecision(release, 0, reason, interval)


def gate_classification(h: int, prediction: int, prediction_set: list[int]) -> GateDecision:
    if h:
        return GateDecision(False, 1, "execution_check_failed", prediction_set)
    release = prediction_set == [prediction]
    reason = "singleton_predicted_class" if release else "non_singleton_or_mismatched_set"
    return GateDecision(release, 0, reason, prediction_set)


def gate_statqa(h: int, prediction_set: list[str]) -> GateDecision:
    if h:
        return GateDecision(False, 1, "execution_check_failed", prediction_set)
    release = prediction_set == ["correct"]
    reason = "correct_only" if release else "ambiguous_or_error_set"
    return GateDecision(release, 0, reason, prediction_set)


def execution_h(output: dict[str, Any], required_keys: tuple[str, ...]) -> int:
    if any(key not in output for key in required_keys):
        return 1
    for value in output.values():
        if isinstance(value, float) and not math.isfinite(value):
            return 1
    return 0
