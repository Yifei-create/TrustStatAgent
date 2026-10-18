from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def prepare_california(manifest: dict[str, str]) -> None:
    source = RAW / "california" / "california_housing.csv"
    frame = pd.read_csv(source)
    indices = np.random.default_rng(42).permutation(len(frame))
    n_train = int(len(frame) * 0.50)
    n_calibration = int(len(frame) * 0.15)
    n_validation = int(len(frame) * 0.15)
    split_frames = {
        "california_train": frame.iloc[indices[:n_train]].reset_index(drop=True),
        "california_calibration": frame.iloc[indices[n_train:n_train + n_calibration]].reset_index(drop=True),
        "california_validation": frame.iloc[indices[n_train + n_calibration:n_train + n_calibration + n_validation]].reset_index(drop=True),
        "california_test": frame.iloc[indices[n_train + n_calibration + n_validation:]].reset_index(drop=True),
    }
    for data_id, split in split_frames.items():
        destination = PROCESSED / f"{data_id}.csv"
        split.to_csv(destination, index=False)
        manifest[data_id] = str(destination)
    feature_destination = PROCESSED / "california_test_features.csv"
    split_frames["california_test"].drop(columns=["MedHouseVal"]).to_csv(feature_destination, index=False)
    manifest["california_test_features"] = str(feature_destination)


def prepare_adult(manifest: dict[str, str]) -> None:
    names = RAW / "adult" / "adult.names"
    train = RAW / "adult" / "adult.data"
    test = RAW / "adult" / "adult.test"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income",
    ]
    train_frame = pd.read_csv(train, names=columns, skipinitialspace=True, na_values="?")
    test_frame = pd.read_csv(test, names=columns, skiprows=1, skipinitialspace=True, na_values="?")
    train_frame["income"] = train_frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    test_frame["income"] = test_frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    train_frame = train_frame.dropna().reset_index(drop=True)
    test_frame = test_frame.dropna().reset_index(drop=True)
    fit_frame, holdout_frame = train_test_split(
        train_frame,
        test_size=0.40,
        random_state=42,
        stratify=train_frame["income"],
    )
    calibration_frame, validation_frame = train_test_split(
        holdout_frame,
        test_size=0.50,
        random_state=42,
        stratify=holdout_frame["income"],
    )
    train_path = PROCESSED / "adult_train.csv"
    calibration_path = PROCESSED / "adult_calibration.csv"
    validation_path = PROCESSED / "adult_validation.csv"
    test_path = PROCESSED / "adult_test.csv"
    fit_frame.reset_index(drop=True).to_csv(train_path, index=False)
    calibration_frame.reset_index(drop=True).to_csv(calibration_path, index=False)
    validation_frame.reset_index(drop=True).to_csv(validation_path, index=False)
    test_frame.to_csv(test_path, index=False)
    feature_path = PROCESSED / "adult_test_features.csv"
    test_frame.drop(columns=["income"]).to_csv(feature_path, index=False)
    manifest.update({"adult_train": str(train_path), "adult_calibration": str(calibration_path), "adult_validation": str(validation_path), "adult_test": str(test_path), "adult_test_features": str(feature_path)})


def prepare_carbon(manifest: dict[str, str]) -> None:
    source = next((path for path in (RAW / "carbon_monitor").glob("*.csv")), None)
    if source is None:
        raise FileNotFoundError("Carbon Monitor CSV is missing")
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"country", "sector", "co2", "date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Carbon Monitor columns are missing: {sorted(required - set(frame.columns))}")
    selected = frame[(frame["country"] == "China") & (frame["sector"] == "Total")].copy()
    selected["date"] = pd.to_datetime(selected["date"], format="mixed")
    selected = selected.sort_values("date").drop_duplicates("date")[["date", "co2"]]
    if selected.empty:
        raise ValueError("China/Total series is empty")
    destination = PROCESSED / "carbon_china_total.csv"
    selected.to_csv(destination, index=False)
    manifest["carbon_china_total"] = str(destination)


def prepare_statqa(manifest: dict[str, str]) -> None:
    root = RAW / "statqa_repo"
    if not root.exists():
        raise FileNotFoundError("StatQA checkout is missing")
    official_mini_prompt = root / "Data" / "Integrated Dataset" / "Dataset with Prompt" / "Test Set" / "mini-StatQA for zero-shot-CoT.csv"
    if official_mini_prompt.exists():
        official_mini_truth = root / "StatQA" / "mini-StatQA.json"
        prompt_frame = pd.read_csv(official_mini_prompt)
        truth_records = json.loads(official_mini_truth.read_text(encoding="utf-8"))
        if len(prompt_frame) != len(truth_records):
            raise ValueError("Official StatQA prompt and ground-truth row counts differ")
        prompt_frame["ground_truth"] = [record["ground_truth"] for record in truth_records]
        merged_path = PROCESSED / "statqa_mini_prompt_with_truth.csv"
        prompt_frame.to_csv(merged_path, index=False)
        manifest["statqa_mini"] = str(merged_path)
    candidates = sorted(root.rglob("*.json")) + sorted(root.rglob("*.csv"))
    full = next((path for path in candidates if path.name.lower() == "statqa.json"), None)
    if full is not None:
        manifest["statqa_full"] = str(full)
    if "statqa_mini" not in manifest and "statqa_full" not in manifest:
        raise FileNotFoundError("Could not locate StatQA benchmark files in official checkout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    prepare_california(manifest)
    prepare_adult(manifest)
    prepare_carbon(manifest)
    prepare_statqa(manifest)
    path = PROCESSED / "datasets.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
