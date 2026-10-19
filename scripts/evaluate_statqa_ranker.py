from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from truststatagent.calibration import StatQAScoreCalibrator
from truststatagent.metrics import gate_metrics


METHODS = ("truststatagent", "react", "without_explicit_planning")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid(rows: list[dict]) -> tuple[list[dict], int]:
    failures = sum(row.get("error") is not None for row in rows)
    return [row for row in rows if row.get("error") is None and row.get("official_correct") is not None], failures


def text_of(row: dict) -> str:
    candidate = row.get("candidate") or {}
    columns = " | ".join(str(value).strip().lower() for value in candidate.get("columns", []))
    methods = " | ".join(str(value).strip().lower() for value in candidate.get("methods", []))
    # The public task category plus the generated answer are observable at runtime.
    # The reference answer and correctness label never enter this representation.
    return f"category: {row.get('category', '')}\ncolumns: {columns}\nmethods: {methods}"


def paths(root: Path, dataset: str, tag: str, split: str) -> list[Path]:
    suffix = f"_{tag}" if tag else ""
    return [root / "results" / "raw" / f"statqa_{dataset}_{method}{suffix}_{split}.jsonl" for method in METHODS]


def load_split(root: Path, dataset: str, tag: str, split: str) -> tuple[list[dict], int]:
    rows: list[dict] = []
    failures = 0
    for method, path in zip(METHODS, paths(root, dataset, tag, split)):
        current, count = valid(read_rows(path))
        for row in current:
            row["_source_method"] = method
        rows.extend(current)
        failures += count
    return rows, failures


def select_point(scores: np.ndarray, correct: np.ndarray, target: float, max_error: float) -> tuple[float, dict]:
    candidates = np.unique(np.r_[scores, np.linspace(0.0, 1.0, 201)])
    records = []
    for threshold in candidates:
        release = scores >= threshold
        metrics = gate_metrics(~correct, release)
        feasible = np.isfinite(metrics["error_release_rate"]) and metrics["error_release_rate"] <= max_error
        key = (1 if feasible else 0, -abs(metrics["release_rate"] - target), -metrics["error_release_rate"] if np.isfinite(metrics["error_release_rate"]) else -1.0)
        records.append((feasible, key, float(threshold), metrics))
    feasible_records = [record for record in records if record[0]]
    if feasible_records:
        selected = max(feasible_records, key=lambda record: record[1])
    else:
        # If the target risk is unattainable on validation, choose the safest point.
        selected = min(records, key=lambda record: (record[3]["error_release_rate"], -record[3]["release_rate"]))
    return selected[2], selected[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", default="mini")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-release", type=float, default=0.50)
    parser.add_argument("--max-error-release", type=float, default=0.05)
    parser.add_argument("--alpha-candidates", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40")
    args = parser.parse_args()
    root = Path(args.root)

    scorer_rows, scorer_failures = load_split(root, args.dataset, args.tag, "scorer_calibration")
    conformal_rows, conformal_failures = load_split(root, args.dataset, args.tag, "conformal_calibration")
    validation_rows, validation_failures = load_split(root, args.dataset, args.tag, "validation")
    test_rows, test_failures = load_split(root, args.dataset, args.tag, "test")
    if not scorer_rows or not conformal_rows or not validation_rows or not test_rows:
        raise ValueError("all four StatQA splits must contain valid rows")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_scorer = vectorizer.fit_transform([text_of(row) for row in scorer_rows])
    y_scorer = np.asarray([bool(row["official_correct"]) for row in scorer_rows])
    if np.unique(y_scorer).size < 2:
        raise ValueError("scorer calibration split contains only one correctness class")
    ranker = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    ranker.fit(x_scorer, y_scorer)

    def scores(rows: list[dict]) -> np.ndarray:
        return ranker.predict_proba(vectorizer.transform([text_of(row) for row in rows]))[:, 1]

    conformal_scores = scores(conformal_rows)
    validation_scores = scores(validation_rows)
    test_scores = scores(test_rows)
    conformal_correct = np.asarray([bool(row["official_correct"]) for row in conformal_rows])
    validation_correct = np.asarray([bool(row["official_correct"]) for row in validation_rows])
    test_correct = np.asarray([bool(row["official_correct"]) for row in test_rows])

    candidates = []
    for alpha in (float(value) for value in args.alpha_candidates.split(",")):
        calibrator = StatQAScoreCalibrator.fit(conformal_scores, conformal_correct, alpha)
        release = np.asarray([calibrator.prediction_set(value) == ["correct"] for value in validation_scores])
        metrics = gate_metrics(~validation_correct, release)
        feasible = np.isfinite(metrics["error_release_rate"]) and metrics["error_release_rate"] <= args.max_error_release
        key = (1 if feasible else 0, -abs(metrics["release_rate"] - args.target_release), -metrics["error_release_rate"] if np.isfinite(metrics["error_release_rate"]) else -1.0)
        candidates.append((feasible, key, alpha, calibrator, metrics))
    feasible_candidates = [candidate for candidate in candidates if candidate[0]]
    if feasible_candidates:
        selected = max(feasible_candidates, key=lambda candidate: candidate[1])
    else:
        # Do not trade away safety for the desired release rate when validation is poor.
        selected = min(candidates, key=lambda candidate: (candidate[4]["error_release_rate"], -candidate[4]["release_rate"]))
    _, _, alpha, calibrator, validation_conformal = selected

    conventional_threshold, validation_conventional = select_point(validation_scores, validation_correct, args.target_release, args.max_error_release)
    output_rows = []
    common = {
        "dataset": "StatQA",
        "selected_alpha": alpha,
        "conformal_q": calibrator.q,
        "conventional_threshold": conventional_threshold,
        "validation_conformal": validation_conformal,
        "validation_conventional": validation_conventional,
        "validation_ranker_auc": float(roc_auc_score(validation_correct, validation_scores)) if np.unique(validation_correct).size > 1 else None,
        "scorer_calibration_failures": scorer_failures,
        "conformal_calibration_failures": conformal_failures,
        "validation_failures": validation_failures,
        "test_failures": test_failures,
        "valid_test_n": len(test_rows),
        "ranker": "tfidf_logistic_regression",
    }
    method_rows = {method: [row for row in test_rows if row["_source_method"] == method] for method in METHODS}
    trust_scores = scores(method_rows["truststatagent"])
    react_scores = scores(method_rows["react"])
    trust_correct = np.asarray([bool(row["official_correct"]) for row in method_rows["truststatagent"]])
    react_correct = np.asarray([bool(row["official_correct"]) for row in method_rows["react"]])
    test_releases = [
        ("TrustStatAgent", "truststatagent", np.asarray([calibrator.prediction_set(value) == ["correct"] for value in trust_scores]), trust_correct),
        ("ReAct with Conventional Gating", "react", react_scores >= conventional_threshold, react_correct),
        ("w/o Conformal Calibration", "truststatagent", trust_scores >= 0.5, trust_correct),
        ("w/o Risk Gate", "truststatagent", np.ones(trust_scores.size, dtype=bool), trust_correct),
    ]
    for variant, source_method, release, labels in test_releases:
        row = dict(common)
        row["variant"] = variant
        row["source_method"] = source_method
        row["accuracy"] = float(np.mean(labels))
        row.update(gate_metrics(~labels, release))
        output_rows.append(row)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n", encoding="utf-8")
    trust_accuracy = next(row["accuracy"] for row in output_rows if row["variant"] == "TrustStatAgent")
    print(json.dumps({"output": str(destination), "ranker_auc": common["validation_ranker_auc"], "test_accuracy": trust_accuracy}, ensure_ascii=False))


if __name__ == "__main__":
    main()
