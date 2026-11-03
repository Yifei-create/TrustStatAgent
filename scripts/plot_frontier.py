"""Regenerate the release-frontier figure (Figure 7) from frozen reference-seed artifacts.

Both panels use the reference configuration reported in the paper tables:
the StatQA panel reuses the official split (seed 42 of run_statqa_multiseed.py)
and the Carbon Monitor panel reuses seed 42 with alpha = 0.10 and
delta_fraction = 0.04 from configs/default.yaml. The calibrated curve sweeps
the gate's release criterion with the generator, the scorer, and the
calibration split held fixed; the conventional curve sweeps a raw
validation-style threshold on the uncalibrated score of the same frozen
candidates it gates in the tables (ReAct candidates on StatQA, the unscaled
rolling residual scale on Carbon Monitor). Operating points reported in the
paper tables are overlaid as markers.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from truststatagent.calibration import RollingResidualCalibrator, StatQAScoreCalibrator
from truststatagent.config import load_config, project_path
from truststatagent.data import DatasetCatalog
from truststatagent.numeric_experiments import _carbon_predictions, _carbon_raw_scales

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_statqa_multiseed", ROOT / "scripts" / "run_statqa_multiseed.py"
)
_msm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_msm)


def released_error(correct: np.ndarray, release: np.ndarray) -> float:
    if not np.any(release):
        return float("nan")
    return float(1.0 - np.mean(correct[release]))


def statqa_panel() -> dict:
    pool = _msm.load_pool(ROOT, "mini", "opt_mini_full")
    cached_split = ROOT / "results" / "raw" / "statqa_mini_split_v2_full.json"
    split = _msm.split_for_seed(pool, 42, cached_split)

    def rows(method: str, name: str) -> list[dict]:
        return [pool[method][task_id] for task_id in split[name] if task_id in pool[method]]

    scorer_rows = [row for method in _msm.METHODS for row in rows(method, "scorer_calibration")]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_scorer = vectorizer.fit_transform([_msm.text_of(row) for row in scorer_rows])
    y_scorer = np.asarray([bool(row["official_correct"]) for row in scorer_rows])
    ranker = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    ranker.fit(x_scorer, y_scorer)

    def scores(method: str, name: str) -> np.ndarray:
        return ranker.predict_proba(vectorizer.transform([_msm.text_of(row) for row in rows(method, name)]))[:, 1]

    def labels(method: str, name: str) -> np.ndarray:
        return np.asarray([bool(row["official_correct"]) for row in rows(method, name)], dtype=bool)

    cal_scores = np.asarray([s for method in _msm.METHODS for s in scores(method, "conformal_calibration")])
    cal_correct = np.asarray([c for method in _msm.METHODS for c in labels(method, "conformal_calibration")])

    trust_scores = scores("truststatagent", "test")
    trust_correct = labels("truststatagent", "test")
    react_scores = scores("react", "test")
    react_correct = labels("react", "test")

    calibrated: list[tuple[float, float]] = []
    for alpha in np.linspace(0.002, 0.998, 500):
        calibrator = StatQAScoreCalibrator.fit(cal_scores, cal_correct, float(alpha))
        release = np.asarray([calibrator.prediction_set(float(value)) == ["correct"] for value in trust_scores])
        calibrated.append((float(np.mean(release)), released_error(trust_correct, release)))

    conventional: list[tuple[float, float]] = []
    for threshold in np.concatenate(([1.0 + 1e-9], np.sort(np.unique(react_scores))[::-1])):
        release = react_scores >= threshold
        conventional.append((float(np.mean(release)), released_error(react_correct, release)))

    return {
        "calibrated": np.asarray(calibrated),
        "conventional": np.asarray(conventional),
        "publish_all_trust": (1.0, released_error(trust_correct, np.ones(trust_correct.size, dtype=bool))),
        "publish_all_react": (1.0, released_error(react_correct, np.ones(react_correct.size, dtype=bool))),
    }


def carbon_panel(config: dict) -> dict:
    catalog = DatasetCatalog(project_path(config, "data/processed/datasets.json"))
    frame = catalog.read("carbon_china_total")
    values = frame["co2"].to_numpy(dtype=float)
    n_train = int(len(values) * 0.50)
    n_calibration = int(len(values) * 0.15)
    n_validation = int(len(values) * 0.15)
    test_start, test_end = n_train + n_calibration + n_validation, len(values)
    carbon = config["data"]["carbon"]
    calibration = config["calibration"]
    lag = int(carbon["seasonal_lag"])
    window = int(carbon["rolling_window"])
    alpha = float(calibration["carbon_alpha_candidates"][0])
    selected_fraction = float(
        [row for row in _reference_rows() if row["variant"] == "TrustStatAgent"][0]["selected_delta_fraction"]
    )
    delta = float(np.median(values[:n_train])) * selected_fraction

    test_points, test_q = _carbon_predictions(values, test_start, test_end, lag, window, alpha)
    _, test_raw_scale = _carbon_raw_scales(values, test_start, test_end, lag, window)
    test_values = values[test_start:test_end]
    error = np.abs(test_points - test_values) > delta
    correct = ~error

    calibrated: list[tuple[float, float]] = []
    for kappa in np.concatenate(([0.0], np.sort(np.unique(test_q)), [test_q.max() * 1.01])):
        release = test_q <= kappa
        calibrated.append((float(np.mean(release)), released_error(correct, release)))

    conventional: list[tuple[float, float]] = []
    for threshold in np.concatenate(([0.0], np.sort(np.unique(test_raw_scale)), [test_raw_scale.max() * 1.01])):
        release = test_raw_scale <= threshold
        conventional.append((float(np.mean(release)), released_error(correct, release)))

    return {
        "calibrated": np.asarray(calibrated),
        "conventional": np.asarray(conventional),
        "publish_all": (1.0, released_error(correct, np.ones(correct.size, dtype=bool))),
    }


def _reference_rows() -> list[dict]:
    path = ROOT / "results" / "metrics" / "numeric_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["dataset"].startswith("Carbon") and row["seed"] == 42]


def operating_points() -> dict:
    statqa_rows = [json.loads(line) for line in (ROOT / "results" / "metrics" / "statqa_mini_multiseed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    statqa = {row["variant"]: row for row in statqa_rows if row["seed"] == 42}
    carbon = {row["variant"]: row for row in _reference_rows()}
    return {
        "statqa_trust": (statqa["TrustStatAgent"]["release_rate"], statqa["TrustStatAgent"]["error_release_rate"]),
        "statqa_conventional": (statqa["ReAct with Conventional Gating"]["release_rate"], statqa["ReAct with Conventional Gating"]["error_release_rate"]),
        "statqa_verbalized": (statqa["Verbalized Confidence Gating"]["release_rate"], statqa["Verbalized Confidence Gating"]["error_release_rate"]),
        "carbon_trust": (carbon["TrustStatAgent"]["release_rate"], carbon["TrustStatAgent"]["error_release_rate"]),
        "carbon_conventional": (carbon["ReAct with Conventional Gating"]["release_rate"], carbon["ReAct with Conventional Gating"]["error_release_rate"]),
    }


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    statqa = statqa_panel()
    carbon = carbon_panel(config)
    ops = operating_points()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    c_cal, c_conv = "#1f4e9c", "#7a7a7a"

    def draw(ax, panel, op_trust, op_conv, publish_all, title, ylim, extra_ops=()):
        cal = panel["calibrated"]
        conv = panel["conventional"]
        cal = cal[np.isfinite(cal[:, 1])]
        conv = conv[np.isfinite(conv[:, 1])]
        order = np.argsort(cal[:, 0])
        ax.plot(100 * cal[order, 0], 100 * cal[order, 1], color=c_cal, lw=1.4, label="Calibrated frontier (this work)")
        order = np.argsort(conv[:, 0])
        ax.plot(100 * conv[order, 0], 100 * conv[order, 1], color=c_conv, lw=1.4, ls="--", label="Conventional frontier")
        ax.scatter([100 * publish_all[0]], [100 * publish_all[1]], marker="s", s=22, facecolor="white", edgecolor="black", linewidth=0.8, zorder=5, label="Publish every candidate")
        ax.scatter([100 * op_trust[0]], [100 * op_trust[1]], marker="o", s=32, color=c_cal, zorder=6, edgecolor="white", linewidth=0.6, label="TrustStatAgent operating point")
        ax.scatter([100 * op_conv[0]], [100 * op_conv[1]], marker="o", s=32, color=c_conv, zorder=6, edgecolor="white", linewidth=0.6, label="Conventional operating point")
        for point, marker, colour, label in extra_ops:
            ax.scatter([100 * point[0]], [100 * point[1]], marker=marker, s=34, color=colour, zorder=6, edgecolor="white", linewidth=0.6, label=label)
        ax.set_xlim(0, 102)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Automatic release rate (%)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)

    draw(
        axes[0], statqa, ops["statqa_trust"], ops["statqa_conventional"], statqa["publish_all_trust"],
        "(a) StatQA", (0, 30),
        extra_ops=[(ops["statqa_verbalized"], "^", "#b3560e", "Verbalized-confidence operating point")],
    )
    axes[0].set_ylabel("Released error rate (%)")
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")

    draw(axes[1], carbon, ops["carbon_trust"], ops["carbon_conventional"], carbon["publish_all"], "(b) Carbon Monitor", (0, 9))

    fig.tight_layout()
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "frontier.png", dpi=300)
    fig.savefig(out_dir / "frontier.pdf")
    print(out_dir / "frontier.png")

    checks = []
    cal = statqa["calibrated"]
    cal = cal[np.isfinite(cal[:, 1])]
    verb_x, verb_y = ops["statqa_verbalized"]
    idx = np.argmin(np.abs(cal[:, 0] - verb_x))
    checks.append("StatQA frontier err at release %.3f = %.3f; verbalized op err = %.3f (above frontier: %s)"
                  % (cal[idx, 0], cal[idx, 1], verb_y, verb_y > cal[idx, 1]))
    conv = statqa["conventional"]
    conv = conv[np.isfinite(conv[:, 1])]
    gaps = []
    for x, y in cal:
        j = np.argmin(np.abs(conv[:, 0] - x))
        if abs(conv[j, 0] - x) < 0.01:
            gaps.append(y - conv[j, 1])
    checks.append("StatQA calibrated-minus-conventional err gap: median %.4f, |max| %.4f over %d matched points"
                  % (float(np.median(gaps)), float(np.max(np.abs(gaps))), len(gaps)))
    cx, cy = ops["carbon_conventional"]
    checks.append("Carbon conventional op err %.4f vs publish-all %.4f (worse: %s)" % (cy, carbon["publish_all"][1], cy > carbon["publish_all"][1]))
    tx, ty = ops["carbon_trust"]
    checks.append("Carbon TrustStatAgent op err %.4f vs publish-all %.4f (better: %s)" % (ty, carbon["publish_all"][1], ty < carbon["publish_all"][1]))
    for line in checks:
        print(line)


if __name__ == "__main__":
    main()
