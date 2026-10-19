from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DatasetCatalog


@dataclass
class ModelArtifact:
    model_id: str
    model: Any
    task: str
    target: str | None = None
    features: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


class StatisticalToolRegistry:
    """Authorized deterministic tools shared by every agent variant."""

    def __init__(self, catalog: DatasetCatalog, seed: int = 42):
        self.catalog = catalog
        self.seed = seed
        self.models: dict[str, ModelArtifact] = {}
        self.predictions: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _require_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} must be a non-empty string")
        return value

    def _require_model(self, model_id: str) -> ModelArtifact:
        model_id = self._require_text(model_id, "model_id")
        if model_id not in self.models:
            raise KeyError(f"Unknown model artifact: {model_id}")
        return self.models[model_id]

    def inspect_dataset(self, data_id: str) -> dict[str, Any]:
        data_id = self._require_text(data_id, "data_id")
        return self.catalog.inspect(data_id)

    def fit_regressor(
        self,
        training_data_id: str,
        target_column: str,
        feature_columns: list[str],
        model_name: str = "fixed_regressor",
    ) -> dict[str, Any]:
        training_data_id = self._require_text(training_data_id, "training_data_id")
        target_column = self._require_text(target_column, "target_column")
        if not isinstance(feature_columns, list) or not feature_columns or not all(isinstance(value, str) and value for value in feature_columns):
            raise TypeError("feature_columns must be a non-empty list of strings")
        frame = self.catalog.read(training_data_id)
        missing = [column for column in [target_column, *feature_columns] if column not in frame.columns]
        if missing:
            raise ValueError(f"Required columns are absent: {missing}")
        if model_name != "fixed_regressor":
            raise ValueError(f"Unsupported fixed regression model: {model_name}")
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=self.seed)
        model.fit(frame[feature_columns], frame[target_column])
        model_id = f"regressor_{len(self.models):04d}"
        self.models[model_id] = ModelArtifact(model_id, model, "regression", target_column, tuple(feature_columns), {})
        return {"model_id": model_id, "status": "success"}

    def fit_classifier(
        self,
        training_data_id: str,
        target_column: str,
        feature_columns: list[str],
        model_name: str = "fixed_classifier",
    ) -> dict[str, Any]:
        training_data_id = self._require_text(training_data_id, "training_data_id")
        target_column = self._require_text(target_column, "target_column")
        if not isinstance(feature_columns, list) or not feature_columns or not all(isinstance(value, str) and value for value in feature_columns):
            raise TypeError("feature_columns must be a non-empty list of strings")
        frame = self.catalog.read(training_data_id)
        missing = [column for column in [target_column, *feature_columns] if column not in frame.columns]
        if missing:
            raise ValueError(f"Required columns are absent: {missing}")
        if model_name != "fixed_classifier":
            raise ValueError(f"Unsupported fixed classification model: {model_name}")
        numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
        categorical = [column for column in feature_columns if column not in numeric]
        preprocess = ColumnTransformer(
            [
                ("numeric", Pipeline([("scale", StandardScaler())]), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ],
            remainder="drop",
        )
        model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000))])
        model.fit(frame[feature_columns], frame[target_column])
        model_id = f"classifier_{len(self.models):04d}"
        self.models[model_id] = ModelArtifact(model_id, model, "classification", target_column, tuple(feature_columns), {})
        return {"model_id": model_id, "status": "success"}

    def predict_regressor(self, model_id: str, prediction_data_id: str) -> dict[str, Any]:
        # Test features are label-free by construction.
        artifact = self._require_model(model_id)
        if artifact.task != "regression":
            raise ValueError(f"Model {model_id} is not a regression artifact")
        prediction_data_id = self._require_text(prediction_data_id, "prediction_data_id")
        frame = self.catalog.read(prediction_data_id)
        prediction = artifact.model.predict(frame[list(artifact.features)])
        result = pd.DataFrame({"prediction": np.asarray(prediction, dtype=float)})
        prediction_id = f"prediction_{len(self.predictions):04d}"
        self.predictions[prediction_id] = result
        if prediction_id not in self.predictions or len(self.predictions[prediction_id]) != len(frame):
            raise RuntimeError("Prediction artifact was not stored for the current task")
        return {"prediction_id": prediction_id, "status": "success", "n_rows": len(result)}

    def predict_classifier(self, model_id: str, prediction_data_id: str) -> dict[str, Any]:
        # Test features are label-free by construction.
        artifact = self._require_model(model_id)
        if artifact.task != "classification":
            raise ValueError(f"Model {model_id} is not a classification artifact")
        prediction_data_id = self._require_text(prediction_data_id, "prediction_data_id")
        frame = self.catalog.read(prediction_data_id)
        labels = artifact.model.predict(frame[list(artifact.features)])
        probabilities = artifact.model.predict_proba(frame[list(artifact.features)])
        probability_frame = pd.DataFrame(probabilities, columns=[f"probability_{index}" for index in range(probabilities.shape[1])])
        result = pd.concat([pd.DataFrame({"prediction": labels}), probability_frame], axis=1)
        prediction_id = f"prediction_{len(self.predictions):04d}"
        self.predictions[prediction_id] = result
        return {
            "prediction_id": prediction_id,
            "status": "success",
            "n_rows": len(result),
            "probability_reference": prediction_id,
            "classes": [str(value) for value in artifact.model.classes_],
        }

    def forecast_seasonal_naive(self, series_data_id: str, cutoff_index: int, seasonal_lag: int = 7) -> dict[str, Any]:
        frame = self.catalog.read(series_data_id)
        if "co2" not in frame.columns:
            raise ValueError("Carbon series must contain a co2 column")
        if cutoff_index < seasonal_lag or cutoff_index >= len(frame):
            raise ValueError("cutoff_index does not provide a valid seasonal lag")
        point = float(frame.iloc[cutoff_index - seasonal_lag]["co2"])
        return {"point_prediction": point, "status": "success", "cutoff_index": cutoff_index}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        methods = {
            "inspect_dataset": self.inspect_dataset,
            "fit_regressor": self.fit_regressor,
            "fit_classifier": self.fit_classifier,
            "predict_regressor": self.predict_regressor,
            "predict_classifier": self.predict_classifier,
            "forecast_seasonal_naive": self.forecast_seasonal_naive,
        }
        if name not in methods:
            raise KeyError(f"Tool is not authorized: {name}")
        return methods[name](**arguments)

    def tool_contract(self) -> list[dict[str, Any]]:
        return [
            {"name": "inspect_dataset", "arguments": ["data_id"]},
            {"name": "fit_regressor", "arguments": ["training_data_id", "target_column", "feature_columns"], "fixed_by_python": "fixed_regressor"},
            {"name": "fit_classifier", "arguments": ["training_data_id", "target_column", "feature_columns"], "fixed_by_python": "fixed_classifier"},
            {"name": "predict_regressor", "arguments": ["model_id", "prediction_data_id"]},
            {"name": "predict_classifier", "arguments": ["model_id", "prediction_data_id"]},
            {"name": "forecast_seasonal_naive", "arguments": ["series_data_id", "cutoff_index", "seasonal_lag"]},
        ]
