from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    content: dict[str, Any]
    raw: dict[str, Any]


class LLMClient:
    """Provider-neutral interface for structured LLM responses."""
    def generate_json(self, messages: list[dict[str, str]]) -> LLMResponse:
        raise NotImplementedError


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name.strip(), value)


class DryRunClient(LLMClient):
    def __init__(self):
        self.calls = 0

    def generate_json(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            return LLMResponse(
                content={
                    "candidate": {"columns": [], "methods": [], "p_correct": 0.5},
                    "execution_status": "dry_run",
                },
                raw={"mode": "dry_run", "message_count": len(messages), "calls": self.calls},
            )
        return LLMResponse(
            content={
                "plan": {
                    "task_type": "regression",
                    "target": "MedHouseVal",
                    "selected_variables": ["MedInc"],
                    "method": "fixed_regressor",
                    "preprocessing": "training-fitted preprocessing",
                    "tool_sequence": ["inspect_dataset", "fit_regressor", "predict_regressor"],
                },
                "tool": "inspect_dataset",
                "arguments": {"data_id": "california_train"},
                "execution_status": "dry_run",
            },
            raw={"mode": "dry_run", "message_count": len(messages), "calls": self.calls},
        )


class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, model: str, api_style: str, api_key_env: str, thinking: str | None, reasoning_effort: str | None, temperature: float, max_tokens: int, timeout: int):
        self.base_url = base_url
        self.model = model
        self.api_style = api_style
        self.api_key_env = api_key_env
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate_json(self, messages: list[dict[str, str]]) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for API-backed runs")
        if self.api_style == "responses":
            payload = {
                "model": self.model,
                "input": messages,
                "max_output_tokens": self.max_tokens,
                "text": {"format": {"type": "json_object"}},
            }
        elif self.api_style == "chat_completions":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self.thinking:
                payload["thinking"] = {"type": self.thinking}
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            raise ValueError(f"Unsupported API style: {self.api_style}")
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"LLM request failed: {error}") from error
        if self.api_style == "responses":
            content = next(
                item["text"]
                for output_item in raw["output"]
                for item in output_item.get("content", [])
                if item.get("type") == "output_text"
            )
        else:
            content = raw["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object")
        return LLMResponse(content=parsed, raw=raw)


def make_client(config: dict[str, Any]) -> LLMClient:
    llm = config["llm"]
    env_file = llm.get("env_file")
    if env_file:
        load_env_file(Path(config["project_root"]) / str(env_file))
    if llm.get("provider") == "dry_run":
        return DryRunClient()
    return OpenAICompatibleClient(
        base_url=str(llm["base_url"]),
        model=str(llm["model"]),
        api_style=str(llm.get("api_style", "chat_completions")),
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        thinking=llm.get("thinking"),
        reasoning_effort=llm.get("reasoning_effort"),
        temperature=float(llm.get("temperature", 0.0)),
        max_tokens=int(llm.get("max_tokens", 1200)),
        timeout=int(llm.get("timeout_seconds", 120)),
    )
