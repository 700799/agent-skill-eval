"""Thin Anthropic SDK wrapper shared by the LLM judge, trigger classifier,
and skill critic.

Everything LLM-shaped goes through one callable signature so tests (and
offline runs) can inject a fake; the real client is imported lazily and only
when actually called.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel

_MAX_TOKENS = 4096


class JsonCaller(Protocol):
    """Callable returning a JSON-object response from a model."""

    def __call__(
        self,
        *,
        system: str,
        user: str,
        model: str,
        schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]: ...


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    raise ValueError(f"no JSON object found in model response: {text[:200]!r}")


def anthropic_call_json(
    *,
    system: str,
    user: str,
    model: str,
    schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """One structured call against the Anthropic API.

    Prefers ``messages.parse`` with the given schema (server-validated JSON);
    falls back to a plain request plus tolerant extraction, with one retry.
    """
    import anthropic

    client = anthropic.Anthropic()
    if schema is not None:
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
            parsed = response.parsed_output
            if isinstance(parsed, BaseModel):
                return parsed.model_dump()
        except Exception:  # noqa: BLE001 - fall back to the plain-JSON path
            pass

    suffix = "\n\nRespond with a single JSON object only — no prose, no code fences."
    last_error: Exception | None = None
    for _ in range(2):
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=system + suffix,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        try:
            return _extract_json_object(text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"model did not return valid JSON after retry: {last_error}")


def default_caller() -> JsonCaller | None:
    """The real caller when credentials are plausibly available, else None.

    The SDK resolves several credential sources itself; probing the client
    constructor (which validates configuration but makes no request) is the
    cheapest reliable check.
    """
    try:
        import anthropic

        anthropic.Anthropic()
    except Exception:  # noqa: BLE001 - any auth/config failure means "offline"
        return None
    return anthropic_call_json
