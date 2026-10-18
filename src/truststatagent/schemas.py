from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str
    question: str
    inputs: dict[str, str]
    allowed_tools: tuple[str, ...]
    task_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisPlan:
    """Structured plan emitted before statistical tool execution."""

    task_type: str
    target: str | None
    selected_variables: tuple[str, ...]
    method: str
    preprocessing: str
    tool_sequence: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AnalysisPlan":
        required = ("task_type", "target", "selected_variables", "method", "preprocessing", "tool_sequence")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Analysis plan is missing fields: {missing}")
        variables = payload["selected_variables"]
        sequence = payload["tool_sequence"]
        if not isinstance(variables, list) or not all(isinstance(value, str) for value in variables):
            raise TypeError("selected_variables must be a list of strings")
        if not isinstance(sequence, list) or not sequence or not all(isinstance(value, str) for value in sequence):
            raise TypeError("tool_sequence must be a non-empty list of strings")
        target = payload["target"]
        if target is not None and not isinstance(target, str):
            raise TypeError("target must be a string or null")
        return cls(str(payload["task_type"]), target, tuple(variables), str(payload["method"]), str(payload["preprocessing"]), tuple(sequence))


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentTrace:
    task_id: str
    method: str
    plan: dict[str, Any] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    candidate: dict[str, Any] | None = None
    execution_status: str = "failed"
    error: str | None = None


@dataclass(frozen=True)
class GateDecision:
    release: bool
    h: int
    reason: str
    conformal_output: Any = None


def task_from_dict(value: dict[str, Any]) -> TaskSpec:
    required = ("task_id", "dataset", "question", "inputs", "allowed_tools")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Task is missing fields: {missing}")
    task_type = str(value.get("task_type", value["dataset"]))
    return TaskSpec(
        task_id=str(value["task_id"]),
        dataset=str(value["dataset"]),
        question=str(value["question"]),
        inputs=dict(value["inputs"]),
        allowed_tools=tuple(str(item) for item in value["allowed_tools"]),
        task_type=task_type,
        metadata=dict(value.get("metadata", {})),
    )
