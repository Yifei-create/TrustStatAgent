from __future__ import annotations

import json
from typing import Any

from .baselines import REACT_SOURCE, ZERO_SHOT_COT_SOURCE
from .schemas import TaskSpec


def zero_shot_cot_prompt(task: TaskSpec) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer the statistical question. Think step by step and return the final answer in JSON."},
        {"role": "user", "content": json.dumps({"question": task.question, "source_protocol": ZERO_SHOT_COT_SOURCE}, ensure_ascii=False)},
    ]


def react_prompt(task: TaskSpec, observations: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Use the ReAct Thought-Action-Observation protocol. Call only authorized statistical tools and return JSON."},
        {"role": "user", "content": json.dumps({"task": task.__dict__, "observations": observations, "source_protocol": REACT_SOURCE}, ensure_ascii=False)},
    ]
