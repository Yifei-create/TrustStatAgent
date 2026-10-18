from __future__ import annotations

import argparse
import json
from pathlib import Path

from truststatagent.tasks import write_tasks
from truststatagent.schemas import TaskSpec


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tasks/tasks.jsonl")
    args = parser.parse_args()
    tasks = [
        TaskSpec(
            task_id="california_housing_0001",
            dataset="california_housing",
            task_type="regression",
            question="Fit the fixed regression model using the authorized training data and predict the authorized test features.",
            inputs={"training_data": "california_train", "prediction_data": "california_test"},
            allowed_tools=("inspect_dataset", "fit_regressor", "predict_regressor"),
        ),
        TaskSpec(
            task_id="adult_0001",
            dataset="adult",
            task_type="classification",
            question="Fit the fixed classifier using the authorized training data and predict the authorized test rows.",
            inputs={"training_data": "adult_train", "prediction_data": "adult_test"},
            allowed_tools=("inspect_dataset", "fit_classifier", "predict_classifier"),
        ),
        TaskSpec(
            task_id="carbon_china_total_0001",
            dataset="carbon_monitor",
            task_type="forecasting",
            question="Forecast the next day of China's total daily CO2 emissions using the fixed seasonal-naive tool.",
            inputs={"series_data": "carbon_china_total"},
            allowed_tools=("inspect_dataset", "forecast_seasonal_naive"),
            metadata={"country": "China", "sector": "Total", "horizon": 1, "seasonal_lag": 7},
        ),
    ]
    output = ROOT / args.output
    write_tasks(tasks, output)
    print(output)


if __name__ == "__main__":
    main()
