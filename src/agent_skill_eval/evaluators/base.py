"""Evaluator protocol, context, and the type-keyed registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_skill_eval.llm import JsonCaller
from agent_skill_eval.models.results import EvalResult
from agent_skill_eval.models.task import TaskSpec
from agent_skill_eval.models.trajectory import Trajectory


@dataclass
class EvalContext:
    task: TaskSpec
    trajectory: Trajectory
    workspace: Path
    skill_name: str | None = None
    llm: JsonCaller | None = None


EvaluatorFn = Callable[[Any, EvalContext], EvalResult]

_REGISTRY: dict[str, EvaluatorFn] = {}


def register(type_name: str) -> Callable[[EvaluatorFn], EvaluatorFn]:
    def decorator(fn: EvaluatorFn) -> EvaluatorFn:
        if type_name in _REGISTRY:
            raise ValueError(f"evaluator type already registered: {type_name}")
        _REGISTRY[type_name] = fn
        return fn

    return decorator


def get_evaluator(type_name: str) -> EvaluatorFn:
    try:
        return _REGISTRY[type_name]
    except KeyError:
        raise KeyError(f"no evaluator registered for type {type_name!r}") from None


def registered_types() -> list[str]:
    return sorted(_REGISTRY)
