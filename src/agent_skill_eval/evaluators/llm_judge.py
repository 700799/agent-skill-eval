"""Rubric-driven LLM judge over the final workspace state and agent output."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent_skill_eval.evaluators.base import EvalContext, register
from agent_skill_eval.models.results import EvalResult
from agent_skill_eval.models.task import LlmJudgeEval

_MAX_CODE_CHARS = 20_000

_SYSTEM = (
    "You are a strict, impartial evaluation judge for coding-agent benchmark runs. "
    "Judge ONLY against the rubric provided; do not reward unrelated qualities. "
    "Score 0.0-1.0 where 1.0 fully satisfies the rubric."
)


class JudgeVerdict(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


def _workspace_code(workspace: Path) -> str:
    chunks: list[str] = []
    budget = _MAX_CODE_CHARS
    for path in sorted(workspace.rglob("*.py")):
        if ".claude" in path.parts or budget <= 0:
            continue
        try:
            content = path.read_text(encoding="utf-8")[:budget]
        except OSError:
            continue
        budget -= len(content)
        chunks.append(f"### {path.relative_to(workspace)}\n```python\n{content}\n```")
    return "\n\n".join(chunks) or "(no Python files in workspace)"


@register("llm_judge")
def evaluate(spec: LlmJudgeEval, ctx: EvalContext) -> EvalResult:
    if ctx.llm is None:
        return EvalResult(
            name="llm_judge",
            passed=False,
            score=0.0,
            required=False,  # skipped evals never gate the trial
            details={"skipped": "no LLM caller configured (offline run or --no-llm)"},
        )
    user = (
        f"## Rubric\n{spec.rubric}\n\n"
        f"## Final workspace code\n{_workspace_code(Path(ctx.workspace))}\n\n"
        f"## Agent's final message\n{ctx.trajectory.final_text() or '(none)'}\n\n"
        "Return JSON: {\"passed\": bool, \"score\": float 0..1, \"reasoning\": str}."
    )
    try:
        raw = ctx.llm(system=_SYSTEM, user=user, model=spec.model, schema=JudgeVerdict)
        verdict = JudgeVerdict.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - judge failure is a result, not a crash
        return EvalResult(
            name="llm_judge",
            passed=False,
            score=0.0,
            required=spec.required,
            details={"error": f"judge call failed: {exc}"},
        )
    return EvalResult(
        name="llm_judge",
        passed=verdict.passed and verdict.score >= spec.pass_threshold,
        score=verdict.score,
        required=spec.required,
        details={
            "reasoning": verdict.reasoning,
            "pass_threshold": spec.pass_threshold,
            "model": spec.model,
        },
    )
