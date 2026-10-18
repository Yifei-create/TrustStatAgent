from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["project_root"] = str(Path(config.get("project_root", config_path.parent.parent)).resolve())
    return config


def project_path(config: dict[str, Any], relative: str) -> Path:
    return Path(config["project_root"]) / relative
