from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import TaskSpec, task_from_dict


def load_tasks(path: str | Path, limit: int | None = None) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            tasks.append(task_from_dict(json.loads(line)))
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def write_tasks(tasks: Iterable[TaskSpec], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.__dict__, ensure_ascii=False) + "\n")
