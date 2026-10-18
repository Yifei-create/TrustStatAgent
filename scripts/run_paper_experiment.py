from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "api"], default="dry-run")
    parser.add_argument("--dataset", choices=["full", "mini"], default="full")
    parser.add_argument("--statqa-limit", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    python = sys.executable
    run([python, "scripts/run_numeric_experiments.py", "--config", str(config_path)])
    for method in ["zero_shot_cot", "react", "truststatagent"]:
        for split in ["calibration", "test"]:
            command = [python, "scripts/run_statqa.py", "--config", str(config_path), "--dataset", args.dataset, "--method", method, "--mode", args.mode, "--split", split]
            if args.statqa_limit is not None:
                command.extend(["--limit", str(args.statqa_limit)])
            run(command)
        output = ROOT / "results" / "metrics" / f"statqa_{args.dataset}_{method}.jsonl"
        run([
            python,
            "scripts/evaluate_statqa.py",
            "--calibration", f"results/raw/statqa_{args.dataset}_{method}_calibration.jsonl",
            "--test", f"results/raw/statqa_{args.dataset}_{method}_test.jsonl",
            "--output", str(output),
        ])
    print(ROOT / "results")


if __name__ == "__main__":
    main()
