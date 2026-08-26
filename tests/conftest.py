"""Shared fixtures: fixture paths, trajectory builders, and a scriptable FakeLLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_skill_eval.models.trajectory import (
    AssistantTurn,
    RunOutcome,
    ToolCall,
    ToolResultRec,
    Trajectory,
    Usage,
)

FIXTURES = Path(__file__).parent / "fixtures"
NDJSON = FIXTURES / "ndjson"
SESSIONS = FIXTURES / "sessions"


@pytest.fixture
def success_fixture() -> Path:
    return NDJSON / "success_with_skill.ndjson"


@pytest.fixture
def loopy_fixture() -> Path:
    return NDJSON / "loopy_no_skill.ndjson"


@pytest.fixture
def error_fixture() -> Path:
    return NDJSON / "error_max_turns.ndjson"


@pytest.fixture
def mined_fixture() -> Path:
    return SESSIONS / "mined_session.jsonl"


class TrajectoryBuilder:
    """Compose synthetic trajectories for metric tests without NDJSON noise."""

    def __init__(self) -> None:
        self._turns: list[AssistantTurn] = []
        self._results: dict[str, ToolResultRec] = {}
        self._inter: list[int] = []
        self._counter = 0

    def turn(
        self,
        *calls: tuple[str, dict[str, Any]],
        text: str = "",
        usage: Usage | None = None,
        result_lines: int | None = None,
        inter_chars: int = 0,
        char_len: int = 0,
    ) -> TrajectoryBuilder:
        tool_calls: list[ToolCall] = []
        for name, tool_input in calls:
            self._counter += 1
            call_id = f"tu_{self._counter}"
            tool_calls.append(ToolCall(id=call_id, name=name, input=tool_input))
            if result_lines is not None:
                self._results[call_id] = ToolResultRec(
                    tool_use_id=call_id,
                    content_text="\n".join(f"line {i}" for i in range(result_lines)),
                )
        self._inter.append(inter_chars)
        self._turns.append(
            AssistantTurn(
                index=len(self._turns),
                text=text,
                tool_calls=tool_calls,
                usage=usage,
                char_len=char_len,
            )
        )
        return self

    def build(self, subtype: str = "success") -> Trajectory:
        return Trajectory(
            source="replay",
            turns=self._turns,
            tool_results=self._results,
            inter_turn_char_lens=self._inter,
            outcome=RunOutcome(subtype=subtype, num_turns=len(self._turns)),
        )


@pytest.fixture
def trajectory_builder() -> type[TrajectoryBuilder]:
    return TrajectoryBuilder


class FakeLLM:
    """Scriptable stand-in for llm.call_json: returns queued responses in order."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, str]] = []

    def __call__(self, *, system: str, user: str, model: str) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "model": model})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture
def fake_llm() -> type[FakeLLM]:
    return FakeLLM
