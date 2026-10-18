from __future__ import annotations

from dataclasses import dataclass


ZERO_SHOT_COT_SOURCE = "https://github.com/kojima-takeshi188/zero_shot_cot"
REACT_SOURCE = "https://github.com/ysymyth/ReAct"


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    source_url: str
    uses_tools: bool
    uses_planning: bool
    uses_conformal: bool
    uses_gate: bool


BASELINES = {
    "zero_shot_cot": BaselineSpec("zero_shot_cot", "Official zero-shot chain-of-thought protocol", ZERO_SHOT_COT_SOURCE, False, False, False, False),
    "react": BaselineSpec("react", "ReAct Thought-Action-Observation protocol with shared statistical tools", REACT_SOURCE, True, False, False, False),
    "react_conventional_gate": BaselineSpec("react_conventional_gate", "ReAct with validation-tuned raw confidence gate", REACT_SOURCE, True, False, False, True),
    "truststatagent": BaselineSpec("truststatagent", "Planning, shared tools, conformal calibration, and risk gate", REACT_SOURCE, True, True, True, True),
}
