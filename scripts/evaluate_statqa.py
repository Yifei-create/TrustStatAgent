from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from truststatagent.calibration import StatQAScoreCalibrator
from truststatagent.metrics import gate_metrics


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_rows(rows: list[dict]) -> tuple[list[dict], int]:
    failures = sum(row.get("error") is not None or row.get("scorer_score") is None for row in rows)
    return [row for row in rows if row.get("error") is None and row.get("scorer_score") is not None and row.get("official_correct") is not None], failures


def release_sets(calibrator: StatQAScoreCalibrator, scores: np.ndarray) -> np.ndarray:
    return np.asarray([calibrator.prediction_set(value) == ["correct"] for value in scores], dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-calibration", required=True)
    parser.add_argument("--conformal-calibration", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--alpha-candidates", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40")
    parser.add_argument("--target-release", type=float, default=0.70)
    parser.add_argument("--max-error-release", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-method", default="truststatagent")
    args = parser.parse_args()

    scorer_rows, scorer_failures = valid_rows(read_rows(Path(args.scorer_calibration)))
    conformal_rows, conformal_failures = valid_rows(read_rows(Path(args.conformal_calibration)))
    validation_rows, validation_failures = valid_rows(read_rows(Path(args.validation)))
    test_rows, test_failures = valid_rows(read_rows(Path(args.test)))
    if not scorer_rows or not conformal_rows or not validation_rows or not test_rows:
        raise ValueError("StatQA evaluation requires non-empty valid four-way splits")

    mapper = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    mapper.fit(
        np.asarray([row["scorer_score"] for row in scorer_rows], dtype=float),
        np.asarray([row["official_correct"] for row in scorer_rows], dtype=bool),
    )
    conformal_scores = mapper.predict(np.asarray([row["scorer_score"] for row in conformal_rows], dtype=float))
    validation_scores = mapper.predict(np.asarray([row["scorer_score"] for row in validation_rows], dtype=float))
    test_scores = mapper.predict(np.asarray([row["scorer_score"] for row in test_rows], dtype=float))
    conformal_correct = np.asarray([row["official_correct"] for row in conformal_rows], dtype=bool)
    validation_correct = np.asarray([row["official_correct"] for row in validation_rows], dtype=bool)
    test_correct = np.asarray([row["official_correct"] for row in test_rows], dtype=bool)

    alpha_candidates = [float(value) for value in args.alpha_candidates.split(",")]
    selected = None
    for alpha in alpha_candidates:
        calibrator = StatQAScoreCalibrator.fit(conformal_scores, conformal_correct, alpha)
        release = release_sets(calibrator, validation_scores)
        metrics = gate_metrics(~validation_correct, release)
        error_release = float(metrics["error_release_rate"])
        release_rate = float(metrics["release_rate"])
        feasible = np.isfinite(error_release) and error_release <= args.max_error_release
        key = (1 if feasible else 0, -abs(release_rate - args.target_release), -error_release if np.isfinite(error_release) else -1.0, float(metrics["balanced_accuracy"]))
        if selected is None or key > selected[0]:
            selected = (key, alpha, calibrator, release_rate, error_release)
    _, alpha, calibrator, validation_release_rate, validation_error_release = selected

    test_release = release_sets(calibrator, test_scores)
    conventional_threshold = float(np.quantile(validation_scores, 1.0 - args.target_release))
    conventional_release = validation_scores >= conventional_threshold
    test_conventional_release = test_scores >= conventional_threshold
    raw_release = test_scores >= 0.5
    rows = []
    common = {
        "dataset": "StatQA",
        "source_method": args.source_method,
        "accuracy": float(np.mean(test_correct)),
        "selected_alpha": alpha,
        "conformal_q": calibrator.q,
        "conventional_threshold": conventional_threshold,
        "validation_release_rate": validation_release_rate,
        "validation_error_release_rate": validation_error_release,
        "scorer_calibration_failures": scorer_failures,
        "conformal_calibration_failures": conformal_failures,
        "validation_failures": validation_failures,
        "test_failures": test_failures,
        "valid_test_n": len(test_rows),
    }
    for variant, release in [
        ("TrustStatAgent", test_release),
        ("ReAct with Conventional Gating", test_conventional_release),
        ("w/o Conformal Calibration", raw_release),
        ("w/o Risk Gate", np.ones(len(test_rows), dtype=bool)),
    ]:
        row = dict(common)
        row["variant"] = variant
        row.update(gate_metrics(~test_correct, release))
        rows.append(row)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
