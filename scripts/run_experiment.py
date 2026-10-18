from __future__ import annotations

import argparse
import json
from pathlib import Path

from truststatagent.agent import TrustStatAgent
from truststatagent.config import load_config, project_path
from truststatagent.data import DatasetCatalog
from truststatagent.llm_client import make_client
from truststatagent.schemas import TaskSpec
from truststatagent.tools import StatisticalToolRegistry


def build_task(config: dict[str, object]) -> TaskSpec:
    return TaskSpec(
        task_id="debug_001",
        dataset="california_housing",
        question="Inspect the authorized dataset and return the next permitted tool call.",
        inputs={"training_data": "california_train"},
        allowed_tools=("inspect_dataset",),
        task_type="debug",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "api"], default="dry-run")
    parser.add_argument("--method", choices=["truststatagent", "react", "without_explicit_planning", "zero_shot_cot"], default="truststatagent")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = project_path(config, "data/processed/datasets.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first: {manifest_path}")
    catalog = DatasetCatalog(manifest_path)
    tools = StatisticalToolRegistry(catalog)
    if args.mode == "dry-run":
        config["llm"]["provider"] = "dry_run"
    llm = make_client(config)
    trace = TrustStatAgent(llm, tools).run(build_task(config), method=args.method)
    output = project_path(config, "results/raw/debug_trace.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
