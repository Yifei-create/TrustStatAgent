from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_table(frame: pd.DataFrame, directory: Path, name: str) -> None:
    frame.to_csv(directory / f"{name}.csv", index=False)
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    (directory / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric", required=True)
    parser.add_argument("--statqa-dir", required=True)
    parser.add_argument("--dataset", choices=["full", "mini"], default="full")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    numeric = read_jsonl(Path(args.numeric))
    numeric_frame = pd.DataFrame(numeric)
    statqa_root = Path(args.statqa_dir)
    statqa_dir = statqa_root / "metrics" if (statqa_root / "metrics").exists() else statqa_root
    statqa_raw_dir = statqa_root / "raw" if (statqa_root / "raw").exists() else statqa_root
    statqa_metrics = {
        path.stem.replace(f"statqa_{args.dataset}_", ""): read_jsonl(path)
        for path in statqa_dir.glob(f"statqa_{args.dataset}_*.jsonl")
    }
    statqa_raw = {
        method: read_jsonl(statqa_raw_dir / f"statqa_{args.dataset}_{method}_test.jsonl")
        for method in ["zero_shot_cot", "react", "truststatagent"]
        if (statqa_raw_dir / f"statqa_{args.dataset}_{method}_test.jsonl").exists()
    }
    table1: list[dict] = []
    for method, rows in statqa_raw.items():
        table1.append({"Method": method, "StatQA official accuracy (%)": 100.0 * sum(row["official_correct"] for row in rows) / len(rows)})
    for dataset in ["Adult", "California Housing", "Carbon Monitor China/Total"]:
        subset = numeric_frame[numeric_frame["dataset"] == dataset]
        if subset.empty:
            continue
        metric = "accuracy" if dataset == "Adult" else "mae"
        for variant, label in [("ReAct with Conventional Gating", "ReAct Statistical Agent"), ("TrustStatAgent", "TrustStatAgent")]:
            values = subset.loc[subset["variant"] == variant, metric]
            table1.append({"Method": f"{label} ({dataset})", f"{dataset} {metric}": float(values.mean()) if not values.empty else None})
    write_table(pd.DataFrame(table1), output, "table1_base_performance")

    table2 = []
    for row in numeric:
        if row["variant"] in {"TrustStatAgent", "ReAct with Conventional Gating"}:
            table2.append({"Dataset": row["dataset"], "Release policy": row["variant"], "Recall": row.get("recall"), "Specificity": row.get("specificity"), "Balanced Accuracy": row.get("balanced_accuracy"), "seed": row.get("seed")})
    for method, rows in statqa_metrics.items():
        for row in rows:
            if row["variant"] in {"TrustStatAgent", "ReAct with Conventional Gating"}:
                table2.append({"Dataset": "StatQA", "Release policy": f"{method}:{row['variant']}", "Recall": row.get("recall"), "Specificity": row.get("specificity"), "Balanced Accuracy": row.get("balanced_accuracy"), "seed": None})
    write_table(pd.DataFrame(table2), output, "table2_gating")

    table3 = [row for row in numeric if row["variant"] in {"w/o Conformal Calibration", "w/o Risk Gate", "TrustStatAgent"}]
    write_table(pd.DataFrame(table3), output, "table3_ablation")
    print(output)


if __name__ == "__main__":
    main()
