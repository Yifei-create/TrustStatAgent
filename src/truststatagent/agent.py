from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .llm_client import LLMClient
from .schemas import AnalysisPlan, AgentTrace, TaskSpec
from .tools import StatisticalToolRegistry


SYSTEM_PROMPT = """You are a statistical analysis agent. Return JSON only.
First inspect authorized data, emit a structured plan, then call only the supplied tools.
All model types, hyperparameters, preprocessing, numerical calculations,
predictions, conformal calibration, and publication decisions are fixed or
performed by Python. Do not send model names or hyperparameters unless the tool
contract explicitly requests them. Do not invent values or access test labels.
For a task that requires a candidate answer, return it with p_correct in [0,1].
For numeric tasks, return the Python-generated result reference instead of
inventing numerical predictions.
The plan must contain task_type, target, selected_variables, method,
preprocessing, and a non-empty tool_sequence listing actual tool names."""


class TrustStatAgent:
    """Execute an authorized plan and return an auditable trace."""
    def __init__(self, llm: LLMClient, tools: StatisticalToolRegistry, max_steps: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def _prompt(self, task: TaskSpec, observations: list[dict[str, Any]]) -> str:
        return json.dumps(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "task_type": task.task_type,
                "question": task.question,
                "inputs": task.inputs,
                "metadata": task.metadata,
                "allowed_tools": task.allowed_tools,
                "tool_contract": self.tools.tool_contract(),
                "observations": observations,
                "required_next_response": {
                    "plan": {
                        "task_type": "string",
                        "target": "string or null",
                        "selected_variables": "list[string]",
                        "method": "string",
                        "preprocessing": "string",
                        "tool_sequence": "list[string]",
                    },
                    "tool": "name or null",
                    "arguments": "object",
                    "candidate": "object or null",
                },
            },
            ensure_ascii=False,
        )

    def run(self, task: TaskSpec, method: str = "truststatagent") -> AgentTrace:
        trace = AgentTrace(task_id=task.task_id, method=method)
        observations: list[dict[str, Any]] = []
        if method == "react":
            system_prompt = "Use an interleaved Thought-Action-Observation protocol. Call only authorized tools and return JSON. Do not decide publication."
        elif method == "without_explicit_planning":
            system_prompt = "Call authorized statistical tools directly to complete the task. Return JSON and do not decide publication."
        elif method == "zero_shot_cot":
            system_prompt = "Think step by step and answer in JSON. Do not call tools."
        else:
            system_prompt = SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        for _ in range(self.max_steps):
            messages.append({"role": "user", "content": self._prompt(task, observations)})
            response = self.llm.generate_json(messages)
            payload = response.content
            if payload.get("plan") is not None:
                trace.plan = asdict(AnalysisPlan.from_payload(dict(payload["plan"])))
            if payload.get("candidate") is not None or payload.get("candidate_answer") is not None:
                if method == "truststatagent" and trace.plan is None:
                    raise ValueError("TrustStatAgent completed without a structured analysis plan")
                trace.candidate = dict(payload.get("candidate", {"answer": payload.get("candidate_answer"), "p_correct": payload.get("p_correct")}))
                trace.execution_status = "success"
                return trace
            if method == "zero_shot_cot":
                trace.candidate = dict(payload)
                trace.execution_status = "success"
                return trace
            tool_name = payload.get("tool")
            if tool_name is None:
                if payload.get("plan") is not None:
                    messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
                    observations.append({"plan": trace.plan, "status": "plan_recorded"})
                    continue
                raise ValueError("LLM response has neither a tool call nor a candidate")
            if tool_name not in task.allowed_tools:
                raise PermissionError(f"Tool {tool_name} is not allowed for task {task.task_id}")
            arguments = dict(payload.get("arguments", {}))
            try:
                observation = self.tools.dispatch(str(tool_name), arguments)
            except (KeyError, PermissionError, TypeError, ValueError) as error:
                observation = {"status": "error", "error_type": type(error).__name__, "message": str(error)}
            call_record = {"tool": tool_name, "arguments": arguments, "observation": observation}
            trace.calls.append(call_record)
            observations.append(call_record)
            messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        raise RuntimeError(f"Maximum tool calls exceeded for task {task.task_id}")
