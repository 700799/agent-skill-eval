"""Evaluator registry: importing this package registers all built-in evaluators."""

from agent_skill_eval.evaluators import (  # noqa: F401  (registration side effects)
    ast_assertions,
    llm_judge,
    pydev_static,
    pydev_tests,
    skill_activation,
    trajectory_efficiency,
)
from agent_skill_eval.evaluators.base import (
    EvalContext,
    get_evaluator,
    register,
    registered_types,
)
from agent_skill_eval.models.results import EvalResult
from agent_skill_eval.models.task import TaskSpec


def run_evaluations(task: TaskSpec, ctx: EvalContext) -> list[EvalResult]:
    return [get_evaluator(spec.type)(spec, ctx) for spec in task.evaluations]


__all__ = [
    "EvalContext",
    "get_evaluator",
    "register",
    "registered_types",
    "run_evaluations",
]
