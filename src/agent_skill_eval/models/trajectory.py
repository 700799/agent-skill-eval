"""Normalized trajectory model — the hub shared by live runs, replay fixtures,
and mined session transcripts.

Parsing models are tolerant (``extra="ignore"``) so CLI stream-format drift
does not break the framework; every field access downstream is defensive.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _TolerantModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Usage(_TolerantModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def context_tokens(self) -> int:
        """Everything the model saw as input this message."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.context_tokens + self.output_tokens


class ToolCall(_TolerantModel):
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultRec(_TolerantModel):
    tool_use_id: str
    content_text: str = ""
    is_error: bool = False

    @property
    def line_count(self) -> int:
        return self.content_text.count("\n") + 1 if self.content_text else 0


class AssistantTurn(_TolerantModel):
    index: int
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    char_len: int = 0


class RunOutcome(_TolerantModel):
    subtype: str = "unknown"
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: Usage | None = None
    num_turns: int = 0
    session_id: str | None = None
    duration_seconds: float | None = None
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.subtype == "success"


class Trajectory(_TolerantModel):
    source: Literal["live", "replay", "mined"]
    model: str | None = None
    turns: list[AssistantTurn] = Field(default_factory=list)
    tool_results: dict[str, ToolResultRec] = Field(default_factory=dict)
    inter_turn_char_lens: list[int] = Field(default_factory=list)
    outcome: RunOutcome = Field(default_factory=RunOutcome)
    raw_path: str | None = None

    def iter_tool_calls(self) -> list[ToolCall]:
        return [call for turn in self.turns for call in turn.tool_calls]

    def result_for(self, call: ToolCall) -> ToolResultRec | None:
        return self.tool_results.get(call.id)

    def skill_activations(self) -> list[str]:
        """Names of skills invoked via the Skill tool, in order."""
        names: list[str] = []
        for call in self.iter_tool_calls():
            if call.name == "Skill":
                skill = call.input.get("skill")
                if isinstance(skill, str) and skill:
                    # "plugin:skill" and "path:skill" forms normalize to last segment
                    names.append(skill.split(":")[-1])
        return names

    def final_text(self) -> str:
        for turn in reversed(self.turns):
            if turn.text:
                return turn.text
        return ""
