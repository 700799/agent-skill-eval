"""Token-waste gate: thresholds over the loop-detection metrics."""

from __future__ import annotations

from agent_skill_eval.evaluators.base import EvalContext, register
from agent_skill_eval.metrics import loops
from agent_skill_eval.models.results import EvalResult
from agent_skill_eval.models.task import TrajectoryEfficiencyEval


def _dim_score(actual: int, limit: int) -> float:
    if actual <= limit:
        return 1.0
    return max(0.0, 1.0 - 0.2 * (actual - limit))


@register("trajectory_efficiency")
def evaluate(spec: TrajectoryEfficiencyEval, ctx: EvalContext) -> EvalResult:
    metrics = loops.analyze(
        ctx.trajectory,
        exempt_tools=spec.exempt_tools,
        context_threshold_tokens=spec.max_context_growth_tokens,
    )
    duplicates = metrics.duplicates.duplicate_tool_calls
    re_reads = metrics.re_reads.file_re_reads
    growth_violations = len(metrics.context_growth.violations)
    passed = (
        duplicates <= spec.max_duplicate_tool_calls
        and re_reads <= spec.max_file_re_reads
        and growth_violations == 0
    )
    score = (
        _dim_score(duplicates, spec.max_duplicate_tool_calls)
        + _dim_score(re_reads, spec.max_file_re_reads)
        + _dim_score(growth_violations, 0)
    ) / 3
    return EvalResult(
        name="trajectory_efficiency",
        passed=passed,
        score=score,
        required=spec.required,
        details={
            "duplicate_tool_calls": duplicates,
            "max_duplicate_tool_calls": spec.max_duplicate_tool_calls,
            "worst_offenders": metrics.duplicates.worst_offenders,
            "file_re_reads": re_reads,
            "max_file_re_reads": spec.max_file_re_reads,
            "re_reads_per_path": metrics.re_reads.per_path,
            "max_context_growth_tokens": metrics.context_growth.max_growth,
            "context_growth_limit": spec.max_context_growth_tokens,
            "context_growth_violations": metrics.context_growth.violations,
            "context_growth_estimated": metrics.context_growth.estimated,
        },
    )
