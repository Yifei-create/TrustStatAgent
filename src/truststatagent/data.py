from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetCatalog:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: dict[str, str] = json.load(handle)

    def path(self, data_id: str) -> Path:
        if data_id not in self.manifest:
            raise KeyError(f"Unknown authorized data_id: {data_id}")
        path = Path(self.manifest[data_id])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        return path

    def read(self, data_id: str) -> pd.DataFrame:
        path = self.path(data_id)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def inspect(self, data_id: str) -> dict[str, Any]:
        frame = self.read(data_id)
        return {
            "data_id": data_id,
            "n_rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "missing": {str(column): int(value) for column, value in frame.isna().sum().items()},
        }
