from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/metrics/metrics.jsonl")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "all_metrics.csv", index=False)
    headers = [str(column) for column in frame.columns]
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    markdown.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    (destination / "all_metrics.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
