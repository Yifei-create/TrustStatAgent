"""Multi-seed StatQA evaluation from stored agent outputs.

The LLM answers are frozen artifacts (temperature 0). What varies across seeds
is the evaluation protocol: the assignment of tasks to scorer calibration,
conformal calibration, validation, and test splits. Seed 42 reuses the cached
official split so its results coincide with the single-configuration report;
seeds 43 and 44 reshuffle the assignment with the same per-category ratios.
All computation is offline and deterministic given the stored outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from truststatagent.calibration import StatQAScoreCalibrator
from truststatagent.metrics import gate_metrics


METHODS = ("truststatagent", "react", "without_explicit_planning")
SPLITS = ("scorer_calibration", "conformal_calibration", "validation", "test")
SEEDS = (42, 43, 44, 45, 46, 47, 48)
ALPHA_CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
TARGET_RELEASE = 0.50
MAX_ERROR_RELEASE = 0.05

_spec = importlib.util.spec_from_file_location(
    "evaluate_statqa_ranker", Path(__file__).with_name("evaluate_statqa_ranker.py")
)
_ranker_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ranker_eval)
text_of = _ranker_eval.text_of
select_point = _ranker_eval.select_point


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pool(root: Path, dataset: str, tag: str) -> dict[str, dict[str, dict]]:
    """Return pool[method][task_id] = row, pooled over the four stored splits."""
    pool: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        pool[method] = {}
        for split in SPLITS:
            path = root / "results" / "raw" / f"statqa_{dataset}_{method}_{tag}_{split}.jsonl"
            for row in read_rows(path):
                if row.get("error") is None and row.get("official_correct") is not None:
                    pool[method][row["task_id"]] = row
    return pool


def split_for_seed(pool: dict[str, dict[str, dict]], seed: int, cached_split: Path) -> dict[str, list[str]]:
    if seed == 42:
        split = json.loads(cached_split.read_text(encoding="utf-8"))
        return {name: split[name] for name in SPLITS}
    rng = random.Random(seed)
    task_ids = sorted(pool["truststatagent"])
    by_category: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        by_category[str(pool["truststatagent"][task_id].get("category", "unknown"))].append(task_id)
    split: dict[str, list[str]] = {name: [] for name in SPLITS}
    for category in sorted(by_category):
        ids = rng.sample(by_category[category], len(by_category[category]))
        n = len(ids)
        scorer_end = max(1, int(round(n * 0.20)))
        conformal_end = scorer_end + max(1, int(round(n * 0.20)))
        validation_end = conformal_end + max(1, int(round(n * 0.20)))
        split["scorer_calibration"].extend(ids[:scorer_end])
        split["conformal_calibration"].extend(ids[scorer_end:conformal_end])
        split["validation"].extend(ids[conformal_end:validation_end])
        split["test"].extend(ids[validation_end:])
    return split


def evaluate_seed(pool: dict[str, dict[str, dict]], split: dict[str, list[str]], seed: int, verbalized: dict[str, float] | None) -> list[dict]:
    def rows(method: str, name: str) -> list[dict]:
        return [pool[method][task_id] for task_id in split[name] if task_id in pool[method]]

    scorer_rows = [row for method in METHODS for row in rows(method, "scorer_calibration")]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_scorer = vectorizer.fit_transform([text_of(row) for row in scorer_rows])
    y_scorer = np.asarray([bool(row["official_correct"]) for row in scorer_rows])
    ranker = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    ranker.fit(x_scorer, y_scorer)

    def scores(method: str, name: str) -> np.ndarray:
        return ranker.predict_proba(vectorizer.transform([text_of(row) for row in rows(method, name)]))[:, 1]

    def labels(method: str, name: str) -> np.ndarray:
        return np.asarray([bool(row["official_correct"]) for row in rows(method, name)], dtype=bool)

    conformal_scores = np.asarray([score for method in METHODS for score in scores(method, "conformal_calibration")])
    conformal_correct = np.asarray([label for method in METHODS for label in labels(method, "conformal_calibration")])

    candidates = []
    for alpha in ALPHA_CANDIDATES:
        calibrator = StatQAScoreCalibrator.fit(conformal_scores, conformal_correct, alpha)
        release = np.asarray([calibrator.prediction_set(value) == ["correct"] for value in scores("truststatagent", "validation")])
        metrics = gate_metrics(~labels("truststatagent", "validation"), release)
        feasible = np.isfinite(metrics["error_release_rate"]) and metrics["error_release_rate"] <= MAX_ERROR_RELEASE
        key = (1 if feasible else 0, -abs(metrics["release_rate"] - TARGET_RELEASE), -metrics["error_release_rate"] if np.isfinite(metrics["error_release_rate"]) else -1.0)
        candidates.append((feasible, key, alpha, calibrator))
    feasible_candidates = [candidate for candidate in candidates if candidate[0]]
    if feasible_candidates:
        _, _, alpha, calibrator = max(feasible_candidates, key=lambda candidate: candidate[1])
    else:
        _, _, alpha, calibrator = min(candidates, key=lambda candidate: (candidate[3].q,))

    react_validation_scores = scores("react", "validation")
    react_validation_correct = labels("react", "validation")
    conventional_threshold, _ = select_point(react_validation_scores, react_validation_correct, TARGET_RELEASE, MAX_ERROR_RELEASE)

    trust_test_scores = scores("truststatagent", "test")
    trust_test_correct = labels("truststatagent", "test")
    react_test_scores = scores("react", "test")
    react_test_correct = labels("react", "test")
    trust_test_rows = rows("truststatagent", "test")
    react_test_rows = rows("react", "test")

    releases = [
        ("TrustStatAgent", np.asarray([calibrator.prediction_set(value) == ["correct"] for value in trust_test_scores]), trust_test_correct),
        ("ReAct with Conventional Gating", react_test_scores >= conventional_threshold, react_test_correct),
        ("w/o Conformal Calibration", trust_test_scores >= 0.5, trust_test_correct),
        ("w/o Risk Gate", np.ones(trust_test_scores.size, dtype=bool), trust_test_correct),
        ("Publish every candidate (ReAct)", np.ones(react_test_scores.size, dtype=bool), react_test_correct),
    ]
    validation_trust_correct = labels("truststatagent", "validation")
    ranker_validation_auc = float(roc_auc_score(validation_trust_correct, scores("truststatagent", "validation"))) if np.unique(validation_trust_correct).size > 1 else None
    verbalized_validation_auc = None
    if verbalized is not None:
        verb = np.asarray([verbalized.get(row["task_id"], np.nan) for row in trust_test_rows])
        verb_validation = np.asarray([verbalized.get(row["task_id"], np.nan) for row in rows("truststatagent", "validation")])
        valid_validation = np.isfinite(verb_validation)
        valid_test = np.isfinite(verb)
        if valid_validation.any() and valid_test.any():
            threshold, _ = select_point(verb_validation[valid_validation], validation_trust_correct[valid_validation], TARGET_RELEASE, MAX_ERROR_RELEASE)
            release = np.zeros(verb.size, dtype=bool)
            release[valid_test] = verb[valid_test] >= threshold
            releases.append(("Verbalized Confidence Gating", release, trust_test_correct))
            if np.unique(validation_trust_correct[valid_validation]).size > 1:
                verbalized_validation_auc = float(roc_auc_score(validation_trust_correct[valid_validation], verb_validation[valid_validation]))

    output = []
    for variant, release, correct in releases:
        row = {"dataset": "StatQA", "variant": variant, "seed": seed, "selected_alpha": alpha, "conformal_q": calibrator.q, "accuracy": float(np.mean(correct)), "ranker_validation_auc": ranker_validation_auc, "verbalized_validation_auc": verbalized_validation_auc}
        row.update(gate_metrics(~correct, release))
        if variant == "TrustStatAgent":
            sets = [calibrator.prediction_set(value) for value in trust_test_scores]
            row["coverage"] = float(np.mean([(("correct" in pred) == bool(z)) for pred, z in zip(sets, trust_test_correct)]))
            row["mean_set_size"] = float(np.mean([len(pred) for pred in sets]))
        output.append(row)

    output.append({
        "dataset": "StatQA",
        "variant": "w/o Explicit Planning",
        "seed": seed,
        "accuracy": float(np.mean(labels("without_explicit_planning", "test"))),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", default="mini")
    parser.add_argument("--tag", default="opt_mini_full")
    parser.add_argument("--verbalized", default="", help="Optional verbalized-confidence jsonl from run_statqa_verbalized_confidence.py")
    parser.add_argument("--output", default="results/metrics/statqa_mini_multiseed.jsonl")
    args = parser.parse_args()
    root = Path(args.root)

    pool = load_pool(root, args.dataset, args.tag)
    cached_split = root / "results" / "raw" / f"statqa_{args.dataset}_split_v2_full.json"
    verbalized = None
    if args.verbalized:
        verbalized = {}
        for record in read_rows(Path(args.verbalized)):
            if record.get("verbalized_p_correct") is not None:
                verbalized[record["task_id"]] = float(record["verbalized_p_correct"])

    all_rows: list[dict] = []
    for seed in SEEDS:
        split = split_for_seed(pool, seed, cached_split)
        all_rows.extend(evaluate_seed(pool, split, seed, verbalized))
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
