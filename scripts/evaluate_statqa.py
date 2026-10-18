from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from truststatagent.calibration import StatQAConfidenceCalibrator
from truststatagent.metrics import gate_metrics


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-method", default="truststatagent")
    args = parser.parse_args()
    calibration = read_rows(Path(args.calibration))
    test = read_rows(Path(args.test))
    calibration_failures = sum(row.get("error") is not None or row.get("p_correct") is None for row in calibration)
    test_failures = sum(row.get("error") is not None or row.get("p_correct") is None for row in test)
    calibration = [row for row in calibration if row.get("error") is None and row.get("p_correct") is not None and row.get("official_correct") is not None]
    test = [row for row in test if row.get("error") is None and row.get("p_correct") is not None and row.get("official_correct") is not None]
    if not calibration or not test:
        raise ValueError("StatQA evaluation requires non-empty valid calibration and test rows")
    calibrator = StatQAConfidenceCalibrator.fit(
        np.asarray([row["p_correct"] for row in calibration]),
        np.asarray([row["official_correct"] for row in calibration]),
        args.alpha,
    )
    test_probability = np.asarray([row["p_correct"] for row in test])
    test_correct = np.asarray([row["official_correct"] for row in test], dtype=bool)
    test_sets = [calibrator.prediction_set(value) for value in test_probability]
    full_release = np.asarray([value == ["correct"] for value in test_sets])
    threshold_candidates = np.unique(np.quantile([row["p_correct"] for row in calibration], np.linspace(0, 1, 101)))
    calibration_error = ~np.asarray([row["official_correct"] for row in calibration], dtype=bool)
    threshold_scores = []
    for threshold in threshold_candidates:
        threshold_scores.append((gate_metrics(calibration_error, np.asarray([row["p_correct"] for row in calibration]) >= threshold)["balanced_accuracy"], threshold))
    conventional_threshold = max(threshold_scores, key=lambda value: (value[0], value[1]))[1]
    conventional_release = test_probability >= conventional_threshold
    raw_release = test_probability >= 0.5
    rows = []
    for variant, release in [
        ("TrustStatAgent", full_release),
        ("ReAct with Conventional Gating", conventional_release),
        ("w/o Conformal Calibration", raw_release),
        ("w/o Risk Gate", np.ones(len(test), dtype=bool)),
    ]:
        row = {"dataset": "StatQA", "source_method": args.source_method, "variant": variant, "accuracy": float(np.mean(test_correct)), "conformal_q": calibrator.q, "conventional_threshold": conventional_threshold, "calibration_failures": calibration_failures, "test_failures": test_failures, "valid_test_n": len(test)}
        row.update(gate_metrics(~test_correct, release))
        rows.append(row)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
