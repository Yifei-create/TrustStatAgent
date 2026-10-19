from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from truststatagent.config import load_config, project_path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "api"], default="api")
    parser.add_argument("--dataset", choices=["full", "mini"], default="mini")
    parser.add_argument("--statqa-limit", type=int)
    parser.add_argument("--skip-numeric-agent", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        parser.error(f"config does not exist: {config_path}")
    python = sys.executable
    run_id = datetime.now(timezone.utc).strftime("paper_%Y%m%dT%H%M%SZ")
    config = load_config(config_path)
    run_dir = project_path(config, f"results/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "config_path": str(config_path), "mode": args.mode, "dataset": args.dataset, "model": config.get("llm", {}).get("model"), "status": "running"}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    numeric = [python, "scripts/run_numeric_experiments.py", "--config", str(config_path), "--run-id", run_id]
    if args.skip_numeric_agent:
        numeric.append("--skip-agent")
    try:
        run(numeric)
        for method in ["truststatagent", "react", "without_explicit_planning"]:
            for split in ["scorer_calibration", "conformal_calibration", "validation", "test"]:
                command = [python, "scripts/run_statqa.py", "--config", str(config_path), "--dataset", args.dataset, "--method", method, "--mode", args.mode, "--skip-scorer", "--split", split, "--tag", run_id]
                if args.statqa_limit is not None:
                    command.extend(["--limit", str(args.statqa_limit)])
                run(command)
        evaluate = [python, "scripts/evaluate_statqa_ranker.py", "--dataset", args.dataset, "--tag", run_id, "--output", f"results/runs/{run_id}/metrics/statqa_{args.dataset}.jsonl"]
        run(evaluate)
    except Exception as error:
        manifest.update({"status": "failed", "failure": f"{type(error).__name__}: {error}"})
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    manifest["status"] = "completed"
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ROOT / "results")


if __name__ == "__main__":
    main()
