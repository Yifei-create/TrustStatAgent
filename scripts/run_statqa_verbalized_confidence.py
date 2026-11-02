"""Elicit verbalized confidence for stored StatQA candidates.

This script implements the verbalized-confidence gating baseline: the LLM
rates, in words, the probability that a previously generated candidate answer
is correct. The score is later thresholded on the validation split, which is
the conventional reliability practice that the calibrated gate is compared
against. No framework component is modified; the script only reads stored
TrustStatAgent outputs and writes one record per task.

Results are appended incrementally, so an interrupted run can be resumed by
repeating the same command; task identifiers already present in the output
file are skipped.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from truststatagent.config import load_config, project_path
from truststatagent.llm_client import make_client


SPLITS = ("scorer_calibration", "conformal_calibration", "validation", "test")

SYSTEM_PROMPT = (
    "You are assessing the reliability of a statistical-analysis answer that you "
    "previously produced. Judge whether the selected columns and methods fully "
    "and correctly answer the statistical question. Return JSON only with one "
    "key, p_correct, a probability between 0 and 1 with three decimals. Use the "
    "full range of the scale and do not default to round values."
)


def _load_rows(root: Path, dataset: str, method: str, tag: str) -> list[dict]:
    rows: list[dict] = []
    suffix = f"_{tag}" if tag else ""
    for split in SPLITS:
        path = root / "results" / "raw" / f"statqa_{dataset}_{method}{suffix}_{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_split"] = split
            rows.append(row)
    return rows


def _query_confidence(llm, row: dict) -> dict:
    record = {
        "task_id": row["task_id"],
        "split": row["_split"],
        "category": row.get("category"),
        "official_correct": row.get("official_correct"),
        "verbalized_p_correct": None,
        "error": None,
    }
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": row["question"], "candidate": row["candidate"]},
                    ensure_ascii=False,
                ),
            },
        ]
        response = llm.generate_json(messages).content
        confidence = float(response["p_correct"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"p_correct out of range: {confidence}")
        record["verbalized_p_correct"] = confidence
    except Exception as exception:  # keep the run alive; failed rows are excluded downstream
        record["error"] = f"{type(exception).__name__}: {exception}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["full", "mini"], default="mini")
    parser.add_argument("--method", default="truststatagent")
    parser.add_argument("--tag", default="opt_mini_full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["project_root"])
    output = args.output or f"results/raw/statqa_{args.dataset}_verbalized_confidence_{args.tag}.jsonl"
    output_path = project_path(config, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("verbalized_p_correct") is not None:
                    done.add(record["task_id"])

    rows = _load_rows(root, args.dataset, args.method, args.tag)
    rows = [
        row
        for row in rows
        if row.get("error") is None
        and row.get("official_correct") is not None
        and row["task_id"] not in done
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"to query: {len(rows)} (already done: {len(done)})")
    if not rows:
        return

    llm = make_client(config)
    lock = threading.Lock()
    completed = 0
    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for record in executor.map(lambda row: _query_confidence(llm, row), rows):
                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"progress: {completed}/{len(rows)}", flush=True)
    print(f"wrote {completed} records to {output_path}")


if __name__ == "__main__":
    main()
