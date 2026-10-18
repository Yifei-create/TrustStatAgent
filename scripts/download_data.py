from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.datasets import fetch_california_housing


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)


def download_california() -> dict[str, Any]:
    destination = RAW / "california" / "california_housing.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_california_housing(as_frame=True).frame
    frame.to_csv(destination, index=False)
    return {"dataset": "california_housing", "path": str(destination), "sha256": sha256(destination), "source": "sklearn.fetch_california_housing"}


def download_adult() -> list[dict[str, Any]]:
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    files = [("adult.data", f"{base}/adult.data"), ("adult.test", f"{base}/adult.test"), ("adult.names", f"{base}/adult.names")]
    records = []
    for name, url in files:
        destination = RAW / "adult" / name
        download(url, destination)
        records.append({"dataset": "adult", "name": name, "path": str(destination), "sha256": sha256(destination), "source": url})
    return records


def download_carbon() -> dict[str, Any]:
    source = "https://zenodo.org/records/5611986/files/CM_v2021.8.csv?download=1"
    destination = RAW / "carbon_monitor" / "CM_v2021.8.csv"
    download(source, destination)
    return {"dataset": "carbon_monitor", "path": str(destination), "sha256": sha256(destination), "source": "https://zenodo.org/records/5611986", "version": "2021.8"}


def download_statqa() -> dict[str, Any]:
    destination = RAW / "statqa_repo"
    if destination.exists():
        revision = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        return {"dataset": "statqa", "path": str(destination), "source": "https://github.com/HKUSTDial/StatQA", "revision": revision, "reused": True}
    subprocess.run(["git", "clone", "https://github.com/HKUSTDial/StatQA.git", str(destination)], check=True)
    revision = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return {"dataset": "statqa", "path": str(destination), "source": "https://github.com/HKUSTDial/StatQA", "revision": revision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", choices=["statqa", "california", "adult", "carbon"])
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("Use --all or --dataset")
    selected = [args.dataset] if args.dataset else ["statqa", "california", "adult", "carbon"]
    records: list[dict[str, Any]] = []
    for dataset in selected:
        if dataset == "statqa":
            records.append(download_statqa())
        elif dataset == "california":
            records.append(download_california())
        elif dataset == "adult":
            records.extend(download_adult())
        elif dataset == "carbon":
            records.append(download_carbon())
    manifest = RAW / "download_manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
