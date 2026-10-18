from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def gate_metrics(is_error: np.ndarray, released: np.ndarray) -> dict[str, float]:
    error = np.asarray(is_error, dtype=bool)
    release = np.asarray(released, dtype=bool)
    intercepted = ~release
    tp = int(np.sum(error & intercepted))
    fn = int(np.sum(error & release))
    fp = int(np.sum(~error & intercepted))
    tn = int(np.sum(~error & release))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    balanced = (recall + specificity) / 2.0 if np.isfinite(recall) and np.isfinite(specificity) else float("nan")
    return {
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "release_rate": float(np.mean(release)),
        "manual_review_rate": float(np.mean(intercepted)),
        "error_release_rate": fn / (fn + tn) if fn + tn else float("nan"),
    }


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic,
    replicates: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("Paired bootstrap inputs must be non-empty one-dimensional arrays of equal shape")
    rng = np.random.default_rng(seed)
    observed = float(statistic(a) - statistic(b))
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, a.size, size=a.size)
        samples[index] = float(statistic(a[selected]) - statistic(b[selected]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)
