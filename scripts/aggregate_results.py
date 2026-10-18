from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    numeric = [column for column in ["accuracy", "mae", "recall", "specificity", "balanced_accuracy", "release_rate"] if column in frame.columns]
    grouped = frame.groupby(["dataset", "variant"], dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(str(part) for part in column if str(part) != "") for column in grouped.columns]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(grouped.to_csv(index=False), encoding="utf-8")
    markdown = ["| " + " | ".join(grouped.columns) + " |", "| " + " | ".join("---" for _ in grouped.columns) + " |"]
    markdown.extend("| " + " | ".join(str(value) for value in row) + " |" for row in grouped.itertuples(index=False, name=None))
    destination.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
