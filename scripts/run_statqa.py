from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from truststatagent.agent import SYSTEM_PROMPT
from truststatagent.config import load_config, project_path
from truststatagent.llm_client import make_client
from truststatagent.prompts import react_prompt, zero_shot_cot_prompt
from truststatagent.statqa import load_statqa_rows, official_correct


def _stratified_split(rows, limit: int | None, seed: int, split_path: Path) -> dict[str, list[str]]:
    """Create one shared task split with all five StatQA categories represented."""
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.task.metadata.get("task", "unknown"))].append(row)
    selected = []
    if limit is None:
        selected = list(rows)
    else:
        rng = random.Random(seed)
        names = sorted(groups)
        if limit < len(names):
            raise ValueError(f"limit must be at least the number of task categories ({len(names)})")
        quotas = {name: limit // len(names) for name in names}
        for name in names[: limit % len(names)]:
            quotas[name] += 1
        for name in names:
            selected.extend(rng.sample(groups[name], min(quotas[name], len(groups[name]))))
    by_category = defaultdict(list)
    for row in selected:
        by_category[str(row.task.metadata.get("task", "unknown"))].append(row.task.task_id)
    calibration_ids, test_ids = [], []
    for name in sorted(by_category):
        ids = by_category[name]
        cut = max(1, int(round(len(ids) * 0.20)))
        calibration_ids.extend(ids[:cut])
        test_ids.extend(ids[cut:])
    split = {"calibration": calibration_ids, "test": test_ids, "categories": sorted(by_category)}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def _scorer_client(config: dict):
    scorer_config = dict(config)
    scorer_llm = dict(config["llm"])
    scorer_llm["api_key_env"] = config.get("scorer", {}).get("api_key_env", "OPENAI_SCORER_API_KEY")
    scorer_llm["model"] = config.get("scorer", {}).get("model", scorer_llm.get("model"))
    scorer_config["llm"] = scorer_llm
    return make_client(scorer_config)


def _score_candidate(scorer, question: str, candidate: dict) -> float:
    messages = [
        {"role": "system", "content": "You are a frozen correctness scorer. Return JSON only with one key p_correct in [0,1]. Judge whether the candidate answers the statistical task using only the question and candidate. Do not invent a reference answer and do not use hidden labels."},
        {"role": "user", "content": json.dumps({"question": question, "candidate": candidate}, ensure_ascii=False)},
    ]
    response = scorer.generate_json(messages).content
    value = float(response["p_correct"])
    if not 0.0 <= value <= 1.0:
        raise ValueError("scorer p_correct must be between 0 and 1")
    return value


def _run_row(row, method: str, llm, scorer) -> dict:
    if method == "zero_shot_cot":
        messages = zero_shot_cot_prompt(row.task)
    elif method == "react":
        messages = react_prompt(row.task, [])
    elif method == "without_explicit_planning":
        messages = [
            {"role": "system", "content": "Answer the statistical question in JSON. Do not provide an explicit plan."},
            {"role": "user", "content": row.task.question},
        ]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row.task.question}]
    planning_instruction = " Include a plan object with task_type, target, selected_variables, method, preprocessing, and tool_sequence." if method == "truststatagent" else ""
    messages.append({"role": "user", "content": "Return one JSON object containing columns and methods for official evaluation." + planning_instruction})
    try:
        response = llm.generate_json(messages)
        plan = response.content.get("plan")
        if "candidate" in response.content:
            candidate = dict(response.content["candidate"])
        elif "columns" in response.content and "methods" in response.content:
            candidate = dict(response.content)
        elif scorer is None:
            candidate = {"columns": [], "methods": []}
        else:
            raise ValueError("StatQA LLM response did not contain a candidate object")
        candidate.pop("plan", None)
        candidate.pop("p_correct", None)
        p_correct = _score_candidate(scorer, row.task.question, candidate) if scorer is not None else 0.5
        error = None
    except (TypeError, ValueError, RuntimeError) as exception:
        plan, candidate, p_correct = None, {}, None
        error = f"{type(exception).__name__}: {exception}"
    return {
        "task_id": row.task.task_id,
        "method": method,
        "candidate": candidate,
        "p_correct": p_correct,
        "official_correct": official_correct(candidate, row.ground_truth) if error is None else None,
        "difficulty": row.difficulty,
        "category": row.task.metadata.get("task"),
        "plan": plan,
        "plan_valid": isinstance(plan, dict) and all(key in plan for key in ("task_type", "target", "selected_variables", "method", "preprocessing", "tool_sequence")),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["full", "mini"], default="mini")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--method", choices=["truststatagent", "without_explicit_planning", "zero_shot_cot", "react"], default="truststatagent")
    parser.add_argument("--mode", choices=["dry-run", "api"], default="dry-run")
    parser.add_argument("--split", choices=["all", "calibration", "test"], default="all")
    parser.add_argument("--tag", default="", help="Optional output tag for independent runs.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads(project_path(config, "data/processed/datasets.json").read_text(encoding="utf-8"))
    path = manifest[f"statqa_{args.dataset}"]
    rows = load_statqa_rows(path)
    split_path = project_path(config, f"results/raw/statqa_{args.dataset}_split_{args.limit or 'full'}.json")
    split = _stratified_split(rows, args.limit, 42, split_path)
    row_by_id = {row.task.task_id: row for row in rows}
    selected_ids = split[args.split] if args.split in {"calibration", "test"} else split["calibration"] + split["test"]
    rows = [row_by_id[task_id] for task_id in selected_ids]
    if args.mode == "dry-run":
        config["llm"]["provider"] = "dry_run"
    llm = make_client(config)
    scorer = _scorer_client(config) if args.mode == "api" else None
    suffix = f"_{args.tag}" if args.tag else ""
    output_path = project_path(config, f"results/raw/statqa_{args.dataset}_{args.method}{suffix}_{args.split}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_run_row, row, args.method, llm, scorer) for row in rows]
        results = [future.result() for future in futures]
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
