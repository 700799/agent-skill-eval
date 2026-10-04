"""Tier 0: static critique of a SKILL.md — free, runs across the whole library.

These checks answer "is this skill even well-formed and worth its context
cost?" before a single token is spent running it.
"""

from __future__ import annotations

import re
from pathlib import Path

from skill_eval_kit import config
from skill_eval_kit.models.portfolio import LintFinding, LintReport, SkillMeta
from skill_eval_kit.portfolio.skills import skill_dir

SEVERITY_PENALTY = {"error": 0.25, "warning": 0.10, "info": 0.02}

MIN_DESCRIPTION_CHARS = 40
#: A good description says when to use the skill, not just what it is.
_TRIGGER_PHRASE = re.compile(
    r"(?i)(\buse\b[^.]{0,40}?\bwhen\b|\bwhenever\b|\btriggers?\b|\bfor when\b)"
)
_CODE_FENCE = re.compile(r"```")
_VAGUE_WORDS = re.compile(r"(?i)\b(helpful|various|stuff|things|etc|general purpose|misc)\b")


def lint_skill(meta: SkillMeta) -> LintReport:
    findings: list[LintFinding] = []

    if not meta.description:
        findings.append(
            LintFinding(
                check="missing-description",
                severity="error",
                message=(
                    "no description in frontmatter: the model cannot decide when to "
                    "load this skill"
                ),
            )
        )
    elif len(meta.description) < MIN_DESCRIPTION_CHARS:
        findings.append(
            LintFinding(
                check="short-description",
                severity="warning",
                message=(
                    f"description is {len(meta.description)} chars; too thin to "
                    "discriminate from other skills"
                ),
            )
        )

    if meta.description and not _TRIGGER_PHRASE.search(meta.description):
        findings.append(
            LintFinding(
                check="no-trigger-guidance",
                severity="warning",
                message="description never says *when* to use the skill (no 'use when ...' clause)",
            )
        )

    vague = sorted({m.lower() for m in _VAGUE_WORDS.findall(meta.description)})
    if vague:
        findings.append(
            LintFinding(
                check="vague-description",
                severity="warning",
                message=f"description leans on vague words: {', '.join(vague)}",
            )
        )

    if meta.token_estimate > config.SKILL_TOKENS_ERROR:
        findings.append(
            LintFinding(
                check="token-footprint",
                severity="error",
                message=(
                    f"~{meta.token_estimate} tokens of skill body is paid on every "
                    "activation; split into reference files loaded on demand"
                ),
            )
        )
    elif meta.token_estimate > config.SKILL_TOKENS_WARNING:
        findings.append(
            LintFinding(
                check="token-footprint",
                severity="warning",
                message=f"~{meta.token_estimate} tokens of skill body is paid on every activation",
            )
        )

    base = skill_dir(meta)
    dead = [ref for ref in meta.referenced_paths if not (base / ref).exists()]
    if dead:
        findings.append(
            LintFinding(
                check="dead-reference",
                severity="error",
                message=f"references files that do not exist: {', '.join(dead[:5])}",
            )
        )

    if not _CODE_FENCE.search(meta.body):
        findings.append(
            LintFinding(
                check="no-examples",
                severity="info",
                message="no code examples; concrete examples raise adherence",
            )
        )

    declared = meta.frontmatter.get("name")
    path = Path(meta.path)
    on_disk = path.parent.name if path.name == "SKILL.md" else path.stem
    if isinstance(declared, str) and declared and declared != on_disk:
        findings.append(
            LintFinding(
                check="name-mismatch",
                severity="warning",
                message=f"frontmatter name {declared!r} does not match location {on_disk!r}",
            )
        )

    if not meta.frontmatter:
        findings.append(
            LintFinding(
                check="no-frontmatter",
                severity="error",
                message="no YAML frontmatter block; the skill cannot advertise itself",
            )
        )

    penalty = sum(SEVERITY_PENALTY[f.severity] for f in findings)
    return LintReport(skill=meta.name, findings=findings, score=max(0.0, 1.0 - penalty))


def lint_all(skills: list[SkillMeta]) -> list[LintReport]:
    return [lint_skill(meta) for meta in skills]
