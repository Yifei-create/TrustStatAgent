from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from truststatagent.agent import TrustStatAgent
from truststatagent.config import load_config, project_path
from truststatagent.data import DatasetCatalog
from truststatagent.numeric_experiments import run_numeric
from truststatagent.llm_client import make_client
from truststatagent.tasks import load_tasks
from truststatagent.tools import StatisticalToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-agent", action="store_true", help="Skip the LLM planning preflight (debugging only).")
    parser.add_argument("--reuse-agent-traces", action="store_true", help="Reuse previously verified successful traces.")
    parser.add_argument("--tasks", default="tasks/tasks.jsonl")
    args = parser.parse_args()
    config = load_config(args.config)
    catalog = DatasetCatalog(project_path(config, "data/processed/datasets.json"))
    agent_traces = []
    trace_path = project_path(config, "results/raw/numeric_agent_traces.jsonl")
    if args.reuse_agent_traces and trace_path.exists():
        agent_traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(agent_traces) != 3 or any(item.get("execution_status") != "success" for item in agent_traces):
            raise ValueError("Existing numeric agent traces are incomplete or unsuccessful")
    elif not args.skip_agent:
        agent = TrustStatAgent(make_client(config), StatisticalToolRegistry(catalog))
        tasks = load_tasks(project_path(config, args.tasks))
        for task in tasks:
            trace = agent.run(task, method="truststatagent")
            if trace.execution_status != "success":
                raise RuntimeError(f"Agent preflight failed for {task.task_id}: {trace.error}")
            agent_traces.append(trace.__dict__)
    calibration = config["calibration"]
    carbon = config["data"]["carbon"]
    all_rows = []
    for seed in config["experiments"]["seeds"]:
        rows = run_numeric(
            catalog,
            alpha=float(calibration["alpha"]),
            california_delta=float(calibration["california_delta"]),
            carbon_delta_fraction=float(calibration["carbon_delta_fraction_of_train_median"]),
            carbon_lag=int(carbon["seasonal_lag"]),
            carbon_window=int(carbon["rolling_window"]),
            seed=int(seed),
            tune=True,
            alpha_candidates=[float(value) for value in calibration.get("alpha_candidates", [calibration["alpha"]])],
            adult_alpha_candidates=[float(value) for value in calibration.get("adult_alpha_candidates", calibration.get("alpha_candidates", [calibration["alpha"]]))],
            california_delta_candidates=[float(value) for value in calibration.get("california_delta_candidates", [calibration["california_delta"]])],
            carbon_delta_candidates=[float(value) for value in calibration.get("carbon_delta_fraction_candidates", [calibration["carbon_delta_fraction_of_train_median"]])],
            carbon_alpha_candidates=[float(value) for value in calibration.get("carbon_alpha_candidates", [calibration["alpha"]])],
            carbon_release_min=float(calibration.get("carbon_release_min", 0.50)),
            carbon_release_max=float(calibration.get("carbon_release_max", 0.80)),
            adult_max_error_release=float(calibration.get("adult_max_error_release", 0.03)),
        )
        for row in rows:
            row["seed"] = int(seed)
            row["agent_preflight"] = "success" if agent_traces else "skipped"
            row["agent_trace_reference"] = "results/raw/numeric_agent_traces.jsonl" if agent_traces else None
            all_rows.append(row)
    output = project_path(config, "results/metrics/numeric_metrics.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows) + "\n", encoding="utf-8")
    trace_output = trace_path
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in agent_traces) + ("\n" if agent_traces else ""), encoding="utf-8")
    run_id = datetime.now(timezone.utc).strftime("numeric_%Y%m%dT%H%M%SZ")
    run_dir = project_path(config, f"results/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    revision_file = project_path(config, "data/raw/download_manifest.json")
    revision_hash = hashlib.sha256(revision_file.read_bytes()).hexdigest() if revision_file.exists() else None
    split_sizes = {data_id: catalog.inspect(data_id)["n_rows"] for data_id in catalog.manifest if data_id != "statqa_full" and data_id != "statqa_mini"}
    manifest = {
        "run_id": run_id,
        "config": config,
        "seed_values": [int(seed) for seed in config["experiments"]["seeds"]],
        "dataset_revision_sha256": revision_hash,
        "split_sizes": split_sizes,
        "selected_parameters": [{key: row[key] for key in ("dataset", "selected_alpha", "selected_delta", "selected_delta_fraction") if key in row} for row in all_rows if row.get("variant") == "TrustStatAgent"],
        "model_version": {"llm": config["llm"].get("model"), "regressor": "HistGradientBoostingRegressor", "classifier": "LogisticRegression", "carbon": "lagged persistence with rolling residual calibration"},
        "agent_trace_reference": str(trace_output),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (project_path(config, "results/runs/latest_manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
