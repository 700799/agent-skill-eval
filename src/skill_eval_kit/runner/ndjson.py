"""Parse Claude Code NDJSON event streams into the normalized :class:`Trajectory`.

One parser serves three sources: live ``claude -p --output-format stream-json``
pipes, recorded replay fixtures, and mined session transcripts (whose records
share the same ``{"type": ..., "message": {...}}`` core shape). Unknown event
types and malformed lines are skipped — the stream format is allowed to drift.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from skill_eval_kit.models.trajectory import (
    AssistantTurn,
    RunOutcome,
    ToolCall,
    ToolResultRec,
    Trajectory,
    Usage,
)

Source = Literal["live", "replay", "mined"]


def _flatten_content(content: Any) -> str:
    """Flatten a message/tool_result ``content`` (string or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("thinking") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return ""


def _parse_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    usage = Usage.model_validate(raw)
    if usage.total_tokens == 0:
        return None
    return usage


class _StreamState:
    def __init__(self) -> None:
        self.turns: list[AssistantTurn] = []
        self.tool_results: dict[str, ToolResultRec] = {}
        self.inter_turn_char_lens: list[int] = []
        self.pending_chars = 0
        self.last_message_id: str | None = None
        self.model: str | None = None
        self.cwd: str | None = None
        self.outcome: RunOutcome | None = None
        self.session_ids: set[str] = set()

    def on_assistant(self, event: dict[str, Any]) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        text = _flatten_content(content)
        tool_calls: list[ToolCall] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    block_id = block.get("id")
                    name = block.get("name")
                    if isinstance(block_id, str) and isinstance(name, str):
                        block_input = block.get("input")
                        tool_calls.append(
                            ToolCall(
                                id=block_id,
                                name=name,
                                input=block_input if isinstance(block_input, dict) else {},
                            )
                        )
        usage = _parse_usage(message.get("usage"))
        char_len = len(json.dumps(content, ensure_ascii=False, default=str)) if content else 0
        message_id = message.get("id")

        # Some CLI versions emit one assistant event per content block sharing a
        # message id; merge those into a single logical turn.
        if (
            isinstance(message_id, str)
            and message_id == self.last_message_id
            and self.turns
        ):
            turn = self.turns[-1]
            turn.text = "\n".join(t for t in (turn.text, text) if t)
            turn.tool_calls.extend(tool_calls)
            turn.char_len += char_len
            if usage is not None:
                turn.usage = usage
            return

        self.inter_turn_char_lens.append(self.pending_chars)
        self.pending_chars = 0
        self.turns.append(
            AssistantTurn(
                index=len(self.turns),
                text=text,
                tool_calls=tool_calls,
                usage=usage,
                char_len=char_len,
            )
        )
        self.last_message_id = message_id if isinstance(message_id, str) else None

    def on_user(self, event: dict[str, Any]) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        self.pending_chars += (
            len(json.dumps(content, ensure_ascii=False, default=str)) if content else 0
        )
        self.last_message_id = None
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if isinstance(tool_use_id, str):
                    self.tool_results[tool_use_id] = ToolResultRec(
                        tool_use_id=tool_use_id,
                        content_text=_flatten_content(block.get("content")),
                        is_error=bool(block.get("is_error", False)),
                    )

    def on_system(self, event: dict[str, Any]) -> None:
        model = event.get("model")
        if isinstance(model, str) and model:
            self.model = model
        cwd = event.get("cwd")
        if isinstance(cwd, str) and cwd:
            self.cwd = cwd
        session_id = event.get("session_id")
        if isinstance(session_id, str):
            self.session_ids.add(session_id)

    def on_result(self, event: dict[str, Any]) -> None:
        self.outcome = RunOutcome(
            subtype=str(event.get("subtype", "unknown")),
            stop_reason=event.get("stop_reason"),
            total_cost_usd=event.get("total_cost_usd"),
            usage=_parse_usage(event.get("usage")),
            num_turns=int(event.get("num_turns", 0) or 0),
            session_id=event.get("session_id"),
        )


def parse_events(
    lines: Iterable[str],
    *,
    source: Source,
    raw_path: str | None = None,
) -> Trajectory:
    state = _StreamState()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "assistant":
            state.on_assistant(event)
        elif event_type == "user":
            state.on_user(event)
        elif event_type == "system":
            state.on_system(event)
        elif event_type == "result":
            state.on_result(event)
        # stream_event / summary / anything else: skipped by design

    outcome = state.outcome or RunOutcome(
        subtype="missing_result",
        num_turns=len(state.turns),
        session_id=next(iter(state.session_ids), None),
    )
    return Trajectory(
        source=source,
        model=state.model,
        cwd=state.cwd,
        turns=state.turns,
        tool_results=state.tool_results,
        inter_turn_char_lens=state.inter_turn_char_lens,
        outcome=outcome,
        raw_path=raw_path,
    )


def parse_file(path: str | Path, *, source: Source) -> Trajectory:
    file_path = Path(path)
    with open(file_path, encoding="utf-8") as fh:
        return parse_events(fh, source=source, raw_path=str(file_path))
