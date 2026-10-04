"""Regenerate the checked-in NDJSON/JSONL fixtures deterministically.

Run from the repo root:  python tests/fixtures/make_fixtures.py

The event shapes mirror `claude -p --output-format stream-json --verbose`
output (system / assistant / user / result events) and, for the mined
fixture, the Claude Code session-transcript record shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
NDJSON_DIR = HERE / "ndjson"
SESSIONS_DIR = HERE / "sessions"

BIG_FILE_LINES = 150
BIG_FILE_TEXT = "\n".join(
    f"{i + 1:6d}\tdef helper_{i}(x: int) -> int:  return x + {i}" for i in range(BIG_FILE_LINES)
)
SMALL_FILE_TEXT = "\n".join(
    f"{i + 1:6d}\t{line}"
    for i, line in enumerate(
        [
            "from fastapi import FastAPI",
            "app = FastAPI()",
            '@app.get("/items")',
            'def read_items(): return {"id": 1, "name": "Item"}',
            "",
        ]
    )
)
# Big enough that chars/4 exceeds the 15k tokens/turn threshold on its own.
HUGE_BLOB = "x" * 70_000


def assistant(
    msg_id: str,
    content: list[dict[str, Any]],
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"id": msg_id, "role": "assistant", "content": content}
    if usage is not None:
        message["usage"] = usage
    return {"type": "assistant", "message": message}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(block_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": tool_input}


def tool_result(tool_use_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": content}],
                }
            ],
        },
    }


def write_ndjson(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def make_success_with_skill() -> list[dict[str, Any]]:
    return [
        {
            "type": "system",
            "subtype": "init",
            "model": "claude-sonnet-5",
            "session_id": "sess-success",
            "cwd": "/workspace",
        },
        assistant(
            "msg_s1",
            [
                text("Loading the fastapi-schema skill before touching the code."),
                tool_use("tu_s1", "Skill", {"skill": "fastapi-schema"}),
            ],
            {"input_tokens": 2400, "output_tokens": 60, "cache_creation_input_tokens": 6000},
        ),
        tool_result(
            "tu_s1",
            "Loaded skill fastapi-schema: always define response schemas as Pydantic v2 "
            "models and validate fields with @field_validator (never legacy @validator).",
        ),
        assistant(
            "msg_s2",
            [tool_use("tu_s2", "Read", {"file_path": "/workspace/app.py"})],
            {"input_tokens": 2500, "output_tokens": 45, "cache_read_input_tokens": 8400},
        ),
        tool_result("tu_s2", SMALL_FILE_TEXT),
        assistant(
            "msg_s3",
            [
                tool_use(
                    "tu_s3",
                    "Write",
                    {
                        "file_path": "/workspace/app.py",
                        "content": (
                            "from fastapi import FastAPI\n"
                            "from pydantic import BaseModel, field_validator\n\n"
                            "app = FastAPI()\n\n\n"
                            "class Item(BaseModel):\n"
                            "    id: int\n"
                            "    name: str\n\n"
                            "    @field_validator('name')\n"
                            "    @classmethod\n"
                            "    def name_not_empty(cls, value: str) -> str:\n"
                            "        if not value:\n"
                            "            raise ValueError('name must not be empty')\n"
                            "        return value\n\n\n"
                            '@app.get("/items", response_model=Item)\n'
                            "def read_items() -> Item:\n"
                            "    return Item(id=1, name='Item')\n"
                        ),
                    },
                )
            ],
            {"input_tokens": 2600, "output_tokens": 380, "cache_read_input_tokens": 8500},
        ),
        tool_result("tu_s3", "File written: /workspace/app.py"),
        assistant(
            "msg_s4",
            [
                text(
                    "Done. app.py now returns an Item Pydantic v2 model with a "
                    "@field_validator, per the fastapi-schema skill."
                )
            ],
            {"input_tokens": 2700, "output_tokens": 90, "cache_read_input_tokens": 8900},
        ),
        {
            "type": "result",
            "subtype": "success",
            "result": "Refactored app.py to a Pydantic v2 response model.",
            "total_cost_usd": 0.031,
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 575,
                "cache_read_input_tokens": 25800,
                "cache_creation_input_tokens": 6000,
            },
            "num_turns": 4,
            "session_id": "sess-success",
            "stop_reason": "end_turn",
        },
    ]


def make_loopy_no_skill() -> list[dict[str, Any]]:
    read_big = {"file_path": "/workspace/big_module.py"}
    events: list[dict[str, Any]] = [
        {
            "type": "system",
            "subtype": "init",
            "model": "claude-sonnet-5",
            "session_id": "sess-loopy",
            "cwd": "/workspace",
        },
        assistant("msg_l1", [text("Exploring."), tool_use("tu_l1", "Read", dict(read_big))]),
        tool_result("tu_l1", BIG_FILE_TEXT),
        assistant("msg_l2", [tool_use("tu_l2", "Read", dict(read_big))]),
        tool_result("tu_l2", BIG_FILE_TEXT),
        assistant("msg_l3", [tool_use("tu_l3", "Read", dict(read_big))]),
        tool_result("tu_l3", BIG_FILE_TEXT),
        assistant("msg_l4", [tool_use("tu_l4", "Read", {"file_path": "/workspace/app.py"})]),
        tool_result("tu_l4", SMALL_FILE_TEXT),
        assistant(
            "msg_l5",
            [
                tool_use(
                    "tu_l5",
                    "Edit",
                    {
                        "file_path": "/workspace/big_module.py",
                        "old_string": "def helper_0",
                        "new_string": "def helper_zero",
                    },
                )
            ],
        ),
        tool_result("tu_l5", "Applied 1 edit."),
        assistant("msg_l6", [tool_use("tu_l6", "Read", dict(read_big))]),
        tool_result("tu_l6", BIG_FILE_TEXT + "\n" + HUGE_BLOB),
        assistant(
            "msg_l7",
            [text("Refactored app.py to return a dict with the right keys. Done.")],
        ),
        {
            "type": "result",
            "subtype": "success",
            "result": "Refactored (eventually).",
            "total_cost_usd": 0.089,
            "usage": {
                "input_tokens": 2000,
                "output_tokens": 900,
                "cache_read_input_tokens": 60000,
                "cache_creation_input_tokens": 2500,
            },
            "num_turns": 7,
            "session_id": "sess-loopy",
            "stop_reason": "end_turn",
        },
    ]
    return events


def make_error_max_turns() -> list[dict[str, Any]]:
    return [
        {
            "type": "system",
            "subtype": "init",
            "model": "claude-sonnet-5",
            "session_id": "sess-err",
            "cwd": "/workspace",
        },
        assistant(
            "msg_e1",
            [tool_use("tu_e1", "Read", {"file_path": "/workspace/app.py"})],
            {"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 7000},
        ),
        tool_result("tu_e1", SMALL_FILE_TEXT),
        assistant(
            "msg_e2",
            [text("I still need to inspect a few more things before editing.")],
            {"input_tokens": 950, "output_tokens": 55, "cache_read_input_tokens": 7100},
        ),
        {
            "type": "result",
            "subtype": "error_max_turns",
            "usage": {
                "input_tokens": 900,
                "output_tokens": 400,
                "cache_read_input_tokens": 30000,
                "cache_creation_input_tokens": 0,
            },
            "num_turns": 6,
            "session_id": "sess-err",
        },
    ]


def make_mined_session() -> list[dict[str, Any]]:
    return [
        {"type": "summary", "summary": "Refactor API models", "leafUuid": "leaf-1"},
        {
            "type": "user",
            "sessionId": "mined-1",
            "message": {"role": "user", "content": "Refactor app.py to use Pydantic models."},
        },
        {
            "type": "assistant",
            "sessionId": "mined-1",
            "message": {
                "id": "msg_m1",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    text("Loading the schema skill."),
                    tool_use("tu_m1", "Skill", {"skill": "fastapi-schema"}),
                ],
                "usage": {
                    "input_tokens": 1200,
                    "output_tokens": 80,
                    "cache_read_input_tokens": 5000,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
        {
            "type": "user",
            "sessionId": "mined-1",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_m1",
                        "content": "Loaded skill fastapi-schema.",
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": "mined-1",
            "message": {
                "id": "msg_m2",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [text("Done — models added.")],
                "usage": {
                    "input_tokens": 1400,
                    "output_tokens": 60,
                    "cache_read_input_tokens": 5200,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
    ]


def main() -> None:
    write_ndjson(NDJSON_DIR / "success_with_skill.ndjson", make_success_with_skill())
    write_ndjson(NDJSON_DIR / "loopy_no_skill.ndjson", make_loopy_no_skill())
    write_ndjson(NDJSON_DIR / "error_max_turns.ndjson", make_error_max_turns())
    write_ndjson(SESSIONS_DIR / "mined_session.jsonl", make_mined_session())
    print(f"fixtures written under {HERE}")


if __name__ == "__main__":
    main()
