from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_sample_q(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample conformal order statistic."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Calibration scores must be a non-empty one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    ordered = np.sort(values)
    k = min(values.size, int(np.ceil((values.size + 1) * (1.0 - alpha))))
    return float(ordered[k - 1])


@dataclass(frozen=True)
class CQRCalibrator:
    alpha: float
    q: float

    @classmethod
    def fit(cls, lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> "CQRCalibrator":
        scores = np.maximum(np.asarray(lower) - np.asarray(y), np.asarray(y) - np.asarray(upper))
        return cls(alpha=alpha, q=max(0.0, finite_sample_q(scores, alpha)))

    def interval(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(lower) - self.q, np.asarray(upper) + self.q


@dataclass(frozen=True)
class ClassificationCalibrator:
    alpha: float
    q: float

    @classmethod
    def fit(cls, probabilities: np.ndarray, y: np.ndarray, alpha: float) -> "ClassificationCalibrator":
        probability = np.asarray(probabilities, dtype=float)
        labels = np.asarray(y, dtype=int)
        scores = 1.0 - probability[np.arange(labels.size), labels]
        return cls(alpha=alpha, q=finite_sample_q(scores, alpha))

    def prediction_set(self, probabilities: np.ndarray) -> list[list[int]]:
        probability = np.asarray(probabilities, dtype=float)
        return [[int(index) for index, value in enumerate(row) if 1.0 - value <= self.q] for row in probability]


@dataclass(frozen=True)
class RollingResidualCalibrator:
    alpha: float
    window: int

    def interval(self, point: float, residuals: np.ndarray) -> tuple[float, float, float]:
        history = np.asarray(residuals, dtype=float)
        if history.size < self.window:
            raise ValueError("Insufficient revealed residuals for rolling calibration")
        q = finite_sample_q(history[-self.window :], self.alpha)
        return float(point - q), float(point + q), float(q)


@dataclass(frozen=True)
class StatQAConfidenceCalibrator:
    alpha: float
    q: float

    @classmethod
    def fit(cls, p_correct: np.ndarray, correct: np.ndarray, alpha: float) -> "StatQAConfidenceCalibrator":
        probability = np.asarray(p_correct, dtype=float)
        labels = np.asarray(correct, dtype=bool)
        true_probability = np.where(labels, probability, 1.0 - probability)
        return cls(alpha=alpha, q=finite_sample_q(1.0 - true_probability, alpha))

    def prediction_set(self, p_correct: float) -> list[str]:
        p = float(p_correct)
        result: list[str] = []
        if 1.0 - p <= self.q:
            result.append("correct")
        if p <= self.q:
            result.append("error")
        return result
