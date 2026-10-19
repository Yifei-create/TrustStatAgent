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
    configured_root = config.get("project_root")
    if configured_root:
        candidate = Path(str(configured_root))
        if candidate.is_absolute():
            raise ValueError("project_root must be relative to the repository or omitted")
        root = (config_path.parent.parent / candidate).resolve()
    else:
        root = config_path.parent.parent.resolve()
    config["project_root"] = str(root)
    return config


def project_path(config: dict[str, Any], relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError(f"Path must be relative to project root: {relative}")
    return Path(config["project_root"]) / requested
