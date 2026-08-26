"""Did the agent actually invoke the target skill, or bypass it?"""

from __future__ import annotations

from agent_skill_eval.evaluators.base import EvalContext, register
from agent_skill_eval.models.results import EvalResult
from agent_skill_eval.models.task import SkillActivationEval


@register("skill_activation")
def evaluate(spec: SkillActivationEval, ctx: EvalContext) -> EvalResult:
    activations = ctx.trajectory.skill_activations()
    activated = spec.expected in activations
    return EvalResult(
        name="skill_activation",
        passed=activated,
        score=1.0 if activated else 0.0,
        required=spec.required,
        details={"expected": spec.expected, "activations": activations},
    )
