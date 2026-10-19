from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibration import CQRCalibrator, ClassificationCalibrator, RollingResidualCalibrator
from .data import DatasetCatalog
from .metrics import accuracy, gate_metrics, mae


CALIFORNIA_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
ADULT_FEATURES = ["age", "workclass", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]


def _adult_pipeline(seed: int) -> Pipeline:
    """Build the fixed Adult preprocessing and classifier pipeline."""
    numeric = [column for column in ADULT_FEATURES if column in {"age", "education_num", "capital_gain", "capital_loss", "hours_per_week"}]
    categorical = [column for column in ADULT_FEATURES if column not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=2000, random_state=seed))])


def tune_threshold(scores: np.ndarray, is_error: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 101)))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        metrics = gate_metrics(is_error, values <= threshold)
        key = (float(metrics["balanced_accuracy"]), float(metrics["specificity"]), -float(threshold))
        if best is None or key > best:
            best = key
            selected = float(threshold)
    return selected


def tune_threshold_at_release_rate(scores: np.ndarray, is_error: np.ndarray, target_release_rate: float) -> float:
    values = np.asarray(scores, dtype=float)
    candidates = np.unique(values)
    best: tuple[float, float, float] | None = None
    selected = float(candidates[0])
    for threshold in candidates:
        release = values <= threshold
        metrics = gate_metrics(is_error, release)
        key = (-abs(float(metrics["release_rate"]) - target_release_rate), float(metrics["balanced_accuracy"]), -float(threshold))
        if best is None or key > best:
            best = key
            selected = float(threshold)
    return selected


def _row(dataset: str, variant: str, base_metric: dict[str, float], gate: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": dataset, "variant": variant}
    result.update(base_metric)
    result.update(gate)
    if variant in {"ReAct with Conventional Gating", "w/o Conformal Calibration"}:
        result["coverage"] = None
    release_error = float(gate.get("error_release_rate", float("nan")))
    result["released_correctness"] = 1.0 - release_error if np.isfinite(release_error) else float("nan")
    return result


def tune_california(catalog: DatasetCatalog, alpha_candidates: list[float], delta_candidates: list[float], seed: int) -> tuple[float, float]:
    train = catalog.read("california_train")
    calibration = catalog.read("california_calibration")
    validation = catalog.read("california_validation")
    point_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
    point_model.fit(train[CALIFORNIA_FEATURES], train["MedHouseVal"])
    val_point = point_model.predict(validation[CALIFORNIA_FEATURES])
    y_cal = calibration["MedHouseVal"].to_numpy()
    y_val = validation["MedHouseVal"].to_numpy()
    best: tuple[float, float, float, float] | None = None
    selected = (alpha_candidates[0], delta_candidates[0])
    for alpha in alpha_candidates:
        lower = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2.0, n_estimators=200, random_state=seed)
        upper = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2.0, n_estimators=200, random_state=seed)
        lower.fit(train[CALIFORNIA_FEATURES], train["MedHouseVal"])
        upper.fit(train[CALIFORNIA_FEATURES], train["MedHouseVal"])
        cal_lower = lower.predict(calibration[CALIFORNIA_FEATURES])
        cal_upper = upper.predict(calibration[CALIFORNIA_FEATURES])
        val_lower = lower.predict(validation[CALIFORNIA_FEATURES])
        val_upper = upper.predict(validation[CALIFORNIA_FEATURES])
        q = max(0.0, CQRCalibrator.fit(cal_lower, cal_upper, y_cal, alpha).q)
        lower_c, upper_c = val_lower - q, val_upper + q
        for delta in delta_candidates:
            release = (lower_c >= val_point - delta) & (upper_c <= val_point + delta)
            metrics = gate_metrics(np.abs(val_point - y_val) > delta, release)
            key = (float(metrics["balanced_accuracy"]), float(metrics["release_rate"]), -float(alpha), -float(delta))
            if best is None or key > best:
                best = key
                selected = (float(alpha), float(delta))
    return selected


def tune_adult(catalog: DatasetCatalog, alpha_candidates: list[float], seed: int, max_error_release: float = 0.03) -> float:
    train = catalog.read("adult_train")
    calibration = catalog.read("adult_calibration")
    validation = catalog.read("adult_validation")
    model = _adult_pipeline(seed)
    model.fit(train[ADULT_FEATURES], train["income"])
    classes = list(model.named_steps["classifier"].classes_)
    cal_prob = model.predict_proba(calibration[ADULT_FEATURES])
    val_prob = model.predict_proba(validation[ADULT_FEATURES])
    val_y = np.asarray([classes.index(value) for value in validation["income"]])
    val_pred = np.argmax(val_prob, axis=1)
    best: tuple[float, float, float] | None = None
    selected = float(alpha_candidates[0])
    for alpha in alpha_candidates:
        conformal = ClassificationCalibrator.fit(cal_prob, np.asarray([classes.index(value) for value in calibration["income"]]), alpha)
        sets = conformal.prediction_set(val_prob)
        release = np.asarray([item == [int(pred)] for item, pred in zip(sets, val_pred)])
        metrics = gate_metrics(val_pred != val_y, release)
        error_release = float(metrics["error_release_rate"])
        if np.isfinite(error_release) and error_release <= max_error_release:
            key = (1.0, float(metrics["release_rate"]), float(metrics["balanced_accuracy"]), -float(alpha))
        else:
            key = (0.0, -error_release if np.isfinite(error_release) else -1.0, float(metrics["balanced_accuracy"]), -float(alpha))
        if best is None or key > best:
            best = key
            selected = float(alpha)
    return selected


def tune_carbon(catalog: DatasetCatalog, alpha_candidates: list[float], delta_candidates: list[float], seasonal_lag: int, window: int, release_min: float = 0.50, release_max: float = 0.80) -> tuple[float, float]:
    frame = catalog.read("carbon_china_total")
    values = frame["co2"].to_numpy(dtype=float)
    n_train = int(len(values) * 0.50)
    n_calibration = int(len(values) * 0.15)
    n_validation = int(len(values) * 0.15)
    validation_start = n_train + n_calibration
    validation_end = validation_start + n_validation
    train_median = float(np.median(values[:n_train]))
    val_values = values[validation_start:validation_end]
    best: tuple[float, float, float, float, float] | None = None
    selected = (alpha_candidates[0], delta_candidates[0])
    for alpha in alpha_candidates:
        val_points, val_q = _carbon_predictions(values, validation_start, validation_end, seasonal_lag, window, alpha)
        for fraction in delta_candidates:
            delta = fraction * train_median
            metrics = gate_metrics(np.abs(val_points - val_values) > delta, val_q <= delta)
            error_release = float(metrics["error_release_rate"])
            balanced = float(metrics["balanced_accuracy"])
            if not np.isfinite(balanced):
                balanced = 0.0
            release_rate = float(metrics["release_rate"])
            if np.isfinite(error_release) and error_release <= 0.05 and release_min <= release_rate <= release_max:
                key = (1.0, balanced, release_rate, -float(alpha), -float(fraction))
            else:
                key = (0.0, -abs(release_rate - 0.75), -error_release if np.isfinite(error_release) else -1.0, balanced, -float(alpha))
            if best is None or key > best:
                best = key
                selected = (float(alpha), float(fraction))
    return selected


def evaluate_california(catalog: DatasetCatalog, alpha: float, delta: float, seed: int) -> list[dict[str, Any]]:
    train = catalog.read("california_train")
    calibration = catalog.read("california_calibration")
    validation = catalog.read("california_validation")
    test = catalog.read("california_test")
    point_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
    lower_model = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2.0, n_estimators=200, random_state=seed)
    upper_model = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2.0, n_estimators=200, random_state=seed)
    point_model.fit(train[CALIFORNIA_FEATURES], train["MedHouseVal"])
    lower_model.fit(train[CALIFORNIA_FEATURES], train["MedHouseVal"])
    upper_model.fit(train[CALIFORNIA_FEATURES], train["MedHouseVal"])
    cal_point = point_model.predict(calibration[CALIFORNIA_FEATURES])
    cal_lower = lower_model.predict(calibration[CALIFORNIA_FEATURES])
    cal_upper = upper_model.predict(calibration[CALIFORNIA_FEATURES])
    val_point = point_model.predict(validation[CALIFORNIA_FEATURES])
    val_lower = lower_model.predict(validation[CALIFORNIA_FEATURES])
    val_upper = upper_model.predict(validation[CALIFORNIA_FEATURES])
    test_point = point_model.predict(test[CALIFORNIA_FEATURES])
    test_lower = lower_model.predict(test[CALIFORNIA_FEATURES])
    test_upper = upper_model.predict(test[CALIFORNIA_FEATURES])
    calibrator = CQRCalibrator.fit(cal_lower, cal_upper, calibration["MedHouseVal"].to_numpy(), alpha)
    test_lower_c, test_upper_c = calibrator.interval(test_lower, test_upper)
    val_lower_c, val_upper_c = calibrator.interval(val_lower, val_upper)
    y_test = test["MedHouseVal"].to_numpy()
    error = np.abs(test_point - y_test) > delta
    val_raw_width = np.maximum(val_point - val_lower, val_upper - val_point)
    test_raw_width = np.maximum(test_point - test_lower, test_upper - test_point)
    val_error = np.abs(val_point - validation["MedHouseVal"].to_numpy()) > delta
    raw_threshold = tune_threshold(val_raw_width, val_error)
    full_release = (test_lower_c >= test_point - delta) & (test_upper_c <= test_point + delta)
    conventional_release = test_raw_width <= raw_threshold
    no_gate_release = np.ones(len(test), dtype=bool)
    coverage = float(np.mean((y_test >= test_lower_c) & (y_test <= test_upper_c)))
    base = {
        "mae": mae(y_test, test_point),
        "conformal_q": calibrator.q,
        "coverage": coverage,
        "mean_interval_width": float(np.mean(test_upper_c - test_lower_c)),
        "released_correctness": float(np.mean(~error[full_release])) if np.any(full_release) else float("nan"),
    }
    return [
        _row("California Housing", "TrustStatAgent", base, gate_metrics(error, full_release)),
        _row("California Housing", "ReAct with Conventional Gating", base, gate_metrics(error, conventional_release)),
        _row("California Housing", "w/o Conformal Calibration", base, gate_metrics(error, test_raw_width <= delta)),
        _row("California Housing", "w/o Risk Gate", base, gate_metrics(error, no_gate_release)),
    ]


def evaluate_adult(catalog: DatasetCatalog, alpha: float, seed: int) -> list[dict[str, Any]]:
    train = catalog.read("adult_train")
    calibration = catalog.read("adult_calibration")
    validation = catalog.read("adult_validation")
    test = catalog.read("adult_test")
    model = _adult_pipeline(seed)
    model.fit(train[ADULT_FEATURES], train["income"])
    classes = list(model.named_steps["classifier"].classes_)
    cal_prob = model.predict_proba(calibration[ADULT_FEATURES])
    val_prob = model.predict_proba(validation[ADULT_FEATURES])
    test_prob = model.predict_proba(test[ADULT_FEATURES])
    cal_y = np.asarray([classes.index(value) for value in calibration["income"]])
    val_y = np.asarray([classes.index(value) for value in validation["income"]])
    test_y = np.asarray([classes.index(value) for value in test["income"]])
    conformal = ClassificationCalibrator.fit(cal_prob, cal_y, alpha)
    prediction_sets = conformal.prediction_set(test_prob)
    val_sets = conformal.prediction_set(val_prob)
    test_pred = np.argmax(test_prob, axis=1)
    val_confidence = np.max(val_prob, axis=1)
    val_error = np.argmax(val_prob, axis=1) != val_y
    threshold = tune_threshold(-val_confidence, val_error)
    full_release = np.asarray([item == [int(pred)] for item, pred in zip(prediction_sets, test_pred)])
    conventional_release = -np.max(test_prob, axis=1) <= threshold
    no_gate_release = np.ones(len(test), dtype=bool)
    # Coverage is evaluated against the held-out label, never the model prediction.
    coverage = float(np.mean([int(label) in item for label, item in zip(test_y, prediction_sets)]))
    released_correctness = float(np.mean(test_pred[full_release] == test_y[full_release])) if np.any(full_release) else float("nan")
    base = {"accuracy": accuracy(test_y, test_pred), "conformal_q": conformal.q, "coverage": coverage, "released_correctness": released_correctness}
    error = test_pred != test_y
    return [
        _row("Adult", "TrustStatAgent", base, gate_metrics(error, full_release)),
        _row("Adult", "ReAct with Conventional Gating", base, gate_metrics(error, conventional_release)),
        _row("Adult", "w/o Conformal Calibration", base, gate_metrics(error, np.max(test_prob, axis=1) > 0.5)),
        _row("Adult", "w/o Risk Gate", base, gate_metrics(error, no_gate_release)),
    ]


def _carbon_predictions(values: np.ndarray, start: int, end: int, seasonal_lag: int, window: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    history = [abs(values[index] - values[index - seasonal_lag]) for index in range(seasonal_lag, start)]
    points: list[float] = []
    qs: list[float] = []
    calibrator = RollingResidualCalibrator(alpha=alpha, window=window)
    for index in range(start, end):
        point = float(values[index - seasonal_lag])
        _, _, q = calibrator.interval(point, np.asarray(history))
        points.append(point)
        qs.append(q)
        history.append(abs(values[index] - point))
    return np.asarray(points), np.asarray(qs)


def _carbon_raw_scales(values: np.ndarray, start: int, end: int, seasonal_lag: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    history = [abs(values[index] - values[index - seasonal_lag]) for index in range(seasonal_lag, start)]
    points: list[float] = []
    scales: list[float] = []
    for index in range(start, end):
        point = float(values[index - seasonal_lag])
        scales.append(float(np.mean(history[-window:])))
        points.append(point)
        history.append(abs(values[index] - point))
    return np.asarray(points), np.asarray(scales)


def evaluate_carbon(catalog: DatasetCatalog, alpha: float, delta_fraction: float, seasonal_lag: int, window: int) -> list[dict[str, Any]]:
    frame = catalog.read("carbon_china_total")
    values = frame["co2"].to_numpy(dtype=float)
    n_train = int(len(values) * 0.50)
    n_calibration = int(len(values) * 0.15)
    n_validation = int(len(values) * 0.15)
    calibration_start = n_train
    validation_start = n_train + n_calibration
    test_start, test_end = validation_start + n_validation, len(values)
    train_median = float(np.median(values[:n_train]))
    delta = delta_fraction * train_median
    val_points, val_q = _carbon_predictions(values, validation_start, test_start, seasonal_lag, window, alpha)
    _, val_raw_scale = _carbon_raw_scales(values, validation_start, test_start, seasonal_lag, window)
    val_values = values[validation_start:test_start]
    val_error = np.abs(val_points - val_values) > delta
    conventional_threshold = tune_threshold(val_raw_scale, val_error)
    test_points, test_q = _carbon_predictions(values, test_start, test_end, seasonal_lag, window, alpha)
    _, test_raw_scale = _carbon_raw_scales(values, test_start, test_end, seasonal_lag, window)
    test_values = values[test_start:test_end]
    error = np.abs(test_points - test_values) > delta
    full_release = test_q <= delta
    conventional_release = test_raw_scale <= conventional_threshold
    # Ablation keeps the prespecified raw-score acceptance rule fixed.
    raw_release = test_raw_scale <= delta
    no_calibration_release = np.ones(len(test_values), dtype=bool)
    coverage = float(np.mean((test_values >= test_points - test_q) & (test_values <= test_points + test_q)))
    base = {
        "mae": mae(test_values, test_points),
        "conformal_q_mean": float(np.mean(test_q)),
        "delta": delta,
        "coverage": coverage,
        "mean_interval_width": float(2.0 * np.mean(test_q)),
        "released_correctness": float(np.mean(~error[full_release])) if np.any(full_release) else float("nan"),
    }
    return [
        _row("Carbon Monitor China/Total", "TrustStatAgent", base, gate_metrics(error, full_release)),
        _row("Carbon Monitor China/Total", "ReAct with Conventional Gating", base, gate_metrics(error, conventional_release)),
        _row("Carbon Monitor China/Total", "w/o Conformal Calibration", base, gate_metrics(error, raw_release)),
        _row("Carbon Monitor China/Total", "w/o Risk Gate", base, gate_metrics(error, no_calibration_release)),
    ]


def run_numeric(catalog: DatasetCatalog, alpha: float, california_delta: float, carbon_delta_fraction: float, carbon_lag: int, carbon_window: int, seed: int, tune: bool = True, alpha_candidates: list[float] | None = None, adult_alpha_candidates: list[float] | None = None, california_delta_candidates: list[float] | None = None, carbon_delta_candidates: list[float] | None = None, carbon_alpha_candidates: list[float] | None = None, carbon_release_min: float = 0.50, carbon_release_max: float = 0.80, adult_max_error_release: float = 0.03) -> list[dict[str, Any]]:
    alpha_grid = alpha_candidates or [alpha]
    california_grid = california_delta_candidates or [california_delta]
    carbon_grid = carbon_delta_candidates or [carbon_delta_fraction]
    if tune:
        california_alpha, selected_california_delta = tune_california(catalog, alpha_grid, california_grid, seed)
        adult_alpha = tune_adult(catalog, adult_alpha_candidates or alpha_grid, seed, adult_max_error_release)
        carbon_alpha, selected_carbon_fraction = tune_carbon(catalog, carbon_alpha_candidates or alpha_grid, carbon_grid, carbon_lag, carbon_window, carbon_release_min, carbon_release_max)
    else:
        california_alpha, selected_california_delta = alpha, california_delta
        adult_alpha = alpha
        carbon_alpha, selected_carbon_fraction = alpha, carbon_delta_fraction
    rows = evaluate_california(catalog, california_alpha, selected_california_delta, seed)
    rows.extend(evaluate_adult(catalog, adult_alpha, seed))
    rows.extend(evaluate_carbon(catalog, carbon_alpha, selected_carbon_fraction, carbon_lag, carbon_window))
    for row in rows:
        row["selected_alpha"] = {"California Housing": california_alpha, "Adult": adult_alpha, "Carbon Monitor China/Total": carbon_alpha}.get(row["dataset"])
        if row["dataset"] == "California Housing":
            row["selected_delta"] = selected_california_delta
        if row["dataset"] == "Carbon Monitor China/Total":
            row["selected_delta_fraction"] = selected_carbon_fraction
    return rows
