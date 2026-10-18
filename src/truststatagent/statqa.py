from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import TaskSpec


METHOD_CATALOG = (
    "Correlation Analysis: Pearson Correlation Coefficient, Spearman Correlation Coefficient, Kendall Correlation Coefficient, Partial Correlation Coefficient; "
    "Distribution Compliance Test: Anderson-Darling Test, Shapiro-Wilk Test of Normality, Kolmogorov-Smirnov Test for Normality, Lilliefors Test, Kolmogorov-Smirnov Test, Kolmogorov-Smirnov Test for Uniform distribution, Kolmogorov-Smirnov Test for Gamma distribution, Kolmogorov-Smirnov Test for Exponential distribution; "
    "Contingency Table Test: Chi-square Independence Test, Fisher Exact Test, Mantel-Haenszel Test; "
    "Descriptive Statistics: Mean, Median, Mode, Range, Quartile, Standard Deviation, Skewness, Kurtosis; "
    "Variance Test: Mood Variance Test, Levene Test, Bartlett Test, F-Test for Variance."
)


@dataclass(frozen=True)
class StatQARow:
    task: TaskSpec
    ground_truth: dict[str, list[str]]
    difficulty: str


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("Expected a JSON object string")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _full_prompt(record: dict[str, Any]) -> str:
    return (
        "### Task Description: Select all relevant columns and applicable statistical methods.\n"
        "### Instruction: Return JSON with exactly two keys, columns and methods. "
        "Use only methods in the catalog below.\n"
        f"### Classification List: {METHOD_CATALOG}\n"
        f"### Dataset: {record.get('dataset', '')}\n"
        f"### Statistical Question: {record['refined_question']}\n"
        "### Response:"
    )


def load_statqa_rows(path: str | Path, limit: int | None = None) -> list[StatQARow]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        records = frame.to_dict(orient="records")
    else:
        records = json.loads(source.read_text(encoding="utf-8"))
    rows: list[StatQARow] = []
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        ground_truth = _parse_json(record["ground_truth"])
        question = str(record.get("prompt", _full_prompt(record)))
        task = TaskSpec(
            task_id=f"statqa_{index:06d}",
            dataset="statqa",
            task_type="statqa",
            question=question,
            inputs={},
            allowed_tools=(),
            metadata={"dataset": record.get("dataset"), "task": record.get("task")},
        )
        rows.append(StatQARow(task, {"columns": list(ground_truth.get("columns", [])), "methods": list(ground_truth.get("methods", []))}, str(record.get("difficulty", "unknown"))))
    return rows


def official_correct(candidate: dict[str, Any], ground_truth: dict[str, list[str]]) -> bool:
    columns = candidate.get("columns")
    methods = candidate.get("methods")
    if not isinstance(columns, list) or not isinstance(methods, list):
        return False
    predicted_columns = {str(value).lower().strip() for value in columns}
    predicted_methods = {str(value).lower().strip() for value in methods}
    expected_columns = {str(value).lower().strip() for value in ground_truth["columns"]}
    expected_methods = {str(value).lower().strip() for value in ground_truth["methods"]}
    return predicted_columns == expected_columns and predicted_methods == expected_methods
