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
    scorer_ids, conformal_ids, validation_ids, test_ids = [], [], [], []
    for name in sorted(by_category):
        ids = by_category[name]
        n = len(ids)
        scorer_end = max(1, int(round(n * 0.20)))
        conformal_end = scorer_end + max(1, int(round(n * 0.20)))
        validation_end = conformal_end + max(1, int(round(n * 0.20)))
        scorer_ids.extend(ids[:scorer_end])
        conformal_ids.extend(ids[scorer_end:conformal_end])
        validation_ids.extend(ids[conformal_end:validation_end])
        test_ids.extend(ids[validation_end:])
    split = {"scorer_calibration": scorer_ids, "conformal_calibration": conformal_ids, "validation": validation_ids, "test": test_ids, "categories": sorted(by_category)}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def _scorer_client(config: dict):
    scorer_config = dict(config)
    scorer_llm = dict(config["llm"])
    scorer_llm["api_key_env"] = config.get("scorer", {}).get("api_key_env", "OPENAI_SCORER_API_KEY")
    scorer_llm["model"] = config.get("scorer", {}).get("model", scorer_llm.get("model"))
    scorer_llm["temperature"] = config.get("scorer", {}).get("temperature", scorer_llm.get("temperature", 0.0))
    scorer_config["llm"] = scorer_llm
    client = make_client(scorer_config)
    client.votes = int(config.get("scorer", {}).get("votes", 3))
    return client


def _score_candidate(scorer, question: str, candidate: dict, demonstrations: list[dict], votes: int = 1) -> tuple[float, dict[str, float]]:
    supports: list[float] = []
    for _ in range(votes):
        messages = [
            {"role": "system", "content": "You are an independent frozen correctness judge. Return JSON only with verdict (correct or incorrect), confidence (0 to 1 with three decimals), and one short critique. Check candidate completeness, variable relevance, and statistical-method applicability using only the question and candidate. Do not use hidden labels or invent a reference answer. Do not choose correct without checking for omissions and method conflicts."},
            {"role": "user", "content": json.dumps({"calibration_examples": demonstrations, "question": question, "candidate": candidate}, ensure_ascii=False)},
        ]
        response = scorer.generate_json(messages).content
        verdict = str(response["verdict"]).lower()
        confidence = float(response["confidence"])
        if verdict not in {"correct", "incorrect"} or not 0.0 <= confidence <= 1.0:
            raise ValueError("invalid fixed-judge response")
        supports.append(confidence if verdict == "correct" else 1.0 - confidence)
    score = float(sum(supports) / len(supports))
    return score, {"vote_mean": score, "vote_min": min(supports), "vote_max": max(supports)}


def _messages_for_row(row, method: str, sample_id: int = 0):
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
    planning = " Include a plan object with task_type, target, selected_variables, method, preprocessing, and tool_sequence." if method == "truststatagent" else ""
    checklist = ""
    if method == "truststatagent":
        category = str(row.task.metadata.get("task", "unknown"))
        checklist = (
            f" The public task category is {category}. First map the question to that category, "
            "then select the complete set of relevant columns and applicable methods from the supplied catalog. "
            "Do not add alternative methods merely as hedges, do not omit a required method, and remove duplicates."
        )
    messages.append({"role": "user", "content": f"Solve independently as sampling variant {sample_id}. Do not reuse an unseen answer. Return one JSON object containing columns and methods for official evaluation." + checklist + planning})
    return messages


def _candidate_from_response(response: dict) -> tuple[dict, dict | None]:
    plan = response.get("plan")
    if "candidate" in response:
        candidate = dict(response["candidate"])
    elif "columns" in response and "methods" in response:
        candidate = dict(response)
    else:
        raise ValueError("StatQA LLM response did not contain a candidate object")
    candidate.pop("plan", None)
    candidate.pop("p_correct", None)
    candidate.pop("scorer_score", None)
    return candidate, plan


def _candidate_key(candidate: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = tuple(sorted(str(value).strip().lower() for value in candidate.get("columns", [])))
    methods = tuple(sorted(str(value).strip().lower() for value in candidate.get("methods", [])))
    return columns, methods


def _run_row(row, method: str, llm, scorer, demonstrations: list[dict], consistency_k: int = 1) -> dict:
    try:
        generated = []
        for _ in range(consistency_k):
            response = llm.generate_json(_messages_for_row(row, method, sample_id=_)) .content
            candidate, plan = _candidate_from_response(response)
            generated.append((candidate, plan))
        groups: dict[tuple, list[tuple[dict, dict | None]]] = {}
        for candidate, plan in generated:
            groups.setdefault(_candidate_key(candidate), []).append((candidate, plan))
        majority = max(groups.values(), key=len)
        candidate, plan = majority[0]
        scorer_score = len(majority) / consistency_k
        scorer_components = {"agreement": scorer_score, "candidate_count": float(consistency_k)}
        error = None
    except (TypeError, ValueError, RuntimeError) as exception:
        plan, candidate, scorer_score, scorer_components = None, {}, None, {}
        error = f"{type(exception).__name__}: {exception}"
    return {
        "task_id": row.task.task_id,
        "question": row.task.question,
        "method": method,
        "candidate": candidate,
        "scorer_score": scorer_score,
        "scorer_components": scorer_components,
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
    parser.add_argument("--split", choices=["all", "scorer_calibration", "conformal_calibration", "validation", "test"], default="all")
    parser.add_argument("--tag", default="", help="Optional output tag for independent runs.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--consistency-k", type=int, default=1)
    parser.add_argument("--skip-scorer", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.consistency_k > 1:
        config["llm"]["temperature"] = float(config.get("statqa", {}).get("consistency_temperature", 0.7))
        config["llm"]["api_style"] = "chat_completions"
        config["llm"]["base_url"] = "https://api.openai.com/v1/chat/completions"
    manifest = json.loads(project_path(config, "data/processed/datasets.json").read_text(encoding="utf-8"))
    path = manifest[f"statqa_{args.dataset}"]
    rows = load_statqa_rows(path)
    split_path = project_path(config, f"results/raw/statqa_{args.dataset}_split_v2_{args.limit or 'full'}.json")
    split = _stratified_split(rows, args.limit, 42, split_path)
    row_by_id = {row.task.task_id: row for row in rows}
    selected_ids = split[args.split] if args.split in {"scorer_calibration", "conformal_calibration", "validation", "test"} else split["scorer_calibration"] + split["conformal_calibration"] + split["validation"] + split["test"]
    rows = [row_by_id[task_id] for task_id in selected_ids]
    if args.mode == "dry-run":
        config["llm"]["provider"] = "dry_run"
    llm = make_client(config)
    scorer = _scorer_client(config) if args.mode == "api" and args.consistency_k == 1 and not args.skip_scorer else None
    demonstrations = []
    if scorer is not None and args.split != "scorer_calibration":
        demo_path = project_path(config, f"results/raw/statqa_{args.dataset}_{args.method}{('_' + args.tag) if args.tag else ''}_scorer_calibration.jsonl")
        if demo_path.exists():
            demo_rows = [json.loads(line) for line in demo_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            by_category = defaultdict(list)
            for item in demo_rows:
                by_category[str(item.get("category", "unknown"))].append(item)
            for category in sorted(by_category):
                item = by_category[category][0]
                demonstrations.append({"question": item.get("question", ""), "candidate": item.get("candidate", {}), "correct": bool(item.get("official_correct"))})
    suffix = f"_{args.tag}" if args.tag else ""
    output_path = project_path(config, f"results/raw/statqa_{args.dataset}_{args.method}{suffix}_{args.split}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_run_row, row, args.method, llm, scorer, demonstrations, args.consistency_k) for row in rows]
        results = [future.result() for future in futures]
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
