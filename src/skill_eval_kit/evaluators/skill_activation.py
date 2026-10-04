"""Did the agent actually invoke the target skill, or bypass it?"""

from __future__ import annotations

from skill_eval_kit.evaluators.base import EvalContext, register
from skill_eval_kit.models.results import EvalResult
from skill_eval_kit.models.task import SkillActivationEval


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
