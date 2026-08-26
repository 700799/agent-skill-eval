"""LLM critic: an actionable critique and rewrite plan for a single SKILL.md."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_skill_eval import config
from agent_skill_eval.llm import JsonCaller
from agent_skill_eval.models.portfolio import CriticReport, LintReport, SkillMeta

_MAX_BODY_CHARS = 12_000

_SYSTEM = (
    "You review Agent Skills (SKILL.md files) for a company skill library that has "
    "grown too large. Judge whether this skill earns the context it costs: is the "
    "description specific enough to route to it correctly, are the instructions "
    "concrete and testable, and does it say anything a competent model would not "
    "already do by default? Be blunt — a skill that only restates default behavior "
    "is waste. Score 0.0-1.0 (1.0 = clearly worth keeping as written)."
)


class Critique(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    rewrite_suggestions: list[str] = Field(default_factory=list)
    improved_description: str | None = None


def critique_skill(
    meta: SkillMeta,
    llm: JsonCaller,
    *,
    model: str = config.DEFAULT_CRITIC_MODEL,
    lint: LintReport | None = None,
) -> CriticReport:
    lint_note = ""
    if lint and lint.findings:
        lint_note = "\n\n## Static findings already detected\n" + "\n".join(
            f"- [{f.severity}] {f.check}: {f.message}" for f in lint.findings
        )
    user = (
        f"## Skill name\n{meta.name}\n\n"
        f"## Description (frontmatter)\n{meta.description or '(none)'}\n\n"
        f"## Body (~{meta.token_estimate} tokens)\n{meta.body[:_MAX_BODY_CHARS]}"
        f"{lint_note}\n\n"
        'Return JSON: {"score": float 0..1, "issues": [str], '
        '"rewrite_suggestions": [str], "improved_description": str}.'
    )
    try:
        raw = llm(system=_SYSTEM, user=user, model=model, schema=Critique)
        critique = Critique.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a failed critique is a datum, not a crash
        return CriticReport(
            skill=meta.name,
            score=0.0,
            issues=[f"critic call failed: {exc}"],
        )
    return CriticReport(
        skill=meta.name,
        score=critique.score,
        issues=critique.issues,
        rewrite_suggestions=critique.rewrite_suggestions,
        improved_description=critique.improved_description,
    )


def critique_all(
    skills: list[SkillMeta],
    llm: JsonCaller,
    *,
    model: str = config.DEFAULT_CRITIC_MODEL,
    lint_by_skill: dict[str, LintReport] | None = None,
) -> list[CriticReport]:
    lookup = lint_by_skill or {}
    return [critique_skill(m, llm, model=model, lint=lookup.get(m.name)) for m in skills]
