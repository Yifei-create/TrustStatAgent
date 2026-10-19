from __future__ import annotations
"""Deprecated component runner; use run_paper_experiment.py."""

import argparse
import json
from pathlib import Path

from truststatagent.agent import TrustStatAgent
from truststatagent.config import load_config, project_path
from truststatagent.data import DatasetCatalog
from truststatagent.llm_client import make_client
from truststatagent.tasks import load_tasks
from truststatagent.tools import StatisticalToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="tasks/tasks.jsonl")
    parser.add_argument("--method", choices=["truststatagent", "react", "without_explicit_planning", "zero_shot_cot"], default="truststatagent")
    parser.add_argument("--mode", choices=["dry-run", "api"], default="dry-run")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "dry-run":
        config["llm"]["provider"] = "dry_run"
    catalog = DatasetCatalog(project_path(config, "data/processed/datasets.json"))
    agent = TrustStatAgent(make_client(config), StatisticalToolRegistry(catalog))
    tasks = load_tasks(project_path(config, args.tasks), args.limit)
    output = project_path(config, f"results/raw/agent_{args.method}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            trace = agent.run(task, method=args.method)
            handle.write(json.dumps(trace.__dict__, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
