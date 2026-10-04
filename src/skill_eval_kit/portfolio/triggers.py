"""Tier 1: does each skill fire on the prompts it should — and stay quiet otherwise?

A skill that activates on the wrong tasks is worse than no skill: it burns
its own token footprint on every false positive. Two modes share one output
shape: a cheap `classifier` pass over descriptions, and a `headless` pass
that runs the real CLI and reads Skill tool calls as ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from skill_eval_kit import config
from skill_eval_kit.llm import JsonCaller
from skill_eval_kit.models.portfolio import SkillMeta, TriggerReport, TriggerResult
from skill_eval_kit.models.task import RunLimits, WorkspaceSpec
from skill_eval_kit.runner.base import AgentRunner
from skill_eval_kit.runner.workspace import inject_skill, sandbox

Mode = Literal["classifier", "headless"]

NEGATIVE_ROW = "(negative)"
NO_ACTIVATION = "(none)"

_SYSTEM = (
    "You route developer requests to Agent Skills. Given a request and a catalog of "
    "skills with their descriptions, decide which skills SHOULD activate. Activate a "
    "skill only when its description clearly covers the request; most requests need "
    "none. Return JSON: {\"activations\": [\"skill-name\", ...]} — an empty list is "
    "the right answer when nothing fits."
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbeSpec(_Strict):
    skill: str
    should_trigger: list[str] = Field(min_length=1)
    should_not_trigger: list[str] = Field(default_factory=list)


class ProbeDefaults(_Strict):
    mode: Mode = "classifier"
    classifier_model: str = config.DEFAULT_CLASSIFIER_MODEL


class ProbesFile(_Strict):
    defaults: ProbeDefaults = Field(default_factory=ProbeDefaults)
    skills_dir: str | None = None
    shared_negatives: list[str] = Field(default_factory=list)
    cross_negatives: bool = True
    probes: list[ProbeSpec] = Field(min_length=1)


def load_probes(path: str | Path) -> ProbesFile:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: probes file must contain a YAML mapping")
    return ProbesFile.model_validate(raw)


def _catalog(skills: list[SkillMeta]) -> str:
    return "\n".join(f"- {s.name}: {s.description or '(no description)'}" for s in skills)


def _classify(
    probe: str, skills: list[SkillMeta], llm: JsonCaller, model: str
) -> list[str]:
    raw = llm(
        system=_SYSTEM,
        user=f"## Skill catalog\n{_catalog(skills)}\n\n## Request\n{probe}",
        model=model,
    )
    activations = raw.get("activations", [])
    known = {s.name for s in skills}
    if not isinstance(activations, list):
        return []
    return [a for a in activations if isinstance(a, str) and a in known]


def _headless(
    probe: str, skills: list[SkillMeta], runner: AgentRunner, limits: RunLimits
) -> list[str]:
    with sandbox(WorkspaceSpec()) as workspace:
        for meta in skills:
            inject_skill(workspace, Path(meta.path))
        trajectory = runner.run(probe, workspace, limits, no_skills=False)
    known = {s.name for s in skills}
    return [name for name in trajectory.skill_activations() if name in known]


def _probe_plan(probes: ProbesFile) -> list[tuple[str, str | None]]:
    """(probe text, intended skill) pairs; intended None marks a shared negative."""
    plan: list[tuple[str, str | None]] = []
    for spec in probes.probes:
        plan.extend((text, spec.skill) for text in spec.should_trigger)
    for text in probes.shared_negatives:
        plan.append((text, None))
    return plan


def _negatives_for(skill: str, probes: ProbesFile) -> set[str]:
    negatives: set[str] = set(probes.shared_negatives)
    for spec in probes.probes:
        if spec.skill == skill:
            negatives.update(spec.should_not_trigger)
        elif probes.cross_negatives:
            negatives.update(spec.should_trigger)
    return negatives


def score_results(
    probes: ProbesFile, results: list[TriggerResult], skills: list[SkillMeta]
) -> tuple[list[TriggerReport], dict[str, dict[str, int]]]:
    by_probe = {r.probe: r for r in results}
    reports: list[TriggerReport] = []

    for meta in skills:
        positives = {
            text
            for spec in probes.probes
            if spec.skill == meta.name
            for text in spec.should_trigger
        }
        negatives = _negatives_for(meta.name, probes) - positives
        tp = fn = fp = tn = 0
        skill_results: list[TriggerResult] = []
        for text in sorted(positives | negatives):
            result = by_probe.get(text)
            if result is None:
                continue
            activated = meta.name in result.activated_skills
            if text in positives:
                tp += activated
                fn += not activated
            else:
                fp += activated
                tn += not activated
            skill_results.append(result)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        reports.append(
            TriggerReport(
                skill=meta.name,
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                results=skill_results,
            )
        )

    confusion: dict[str, dict[str, int]] = {}
    for result in results:
        row = result.intended_skill or NEGATIVE_ROW
        bucket = confusion.setdefault(row, {})
        for activated_name in result.activated_skills or [NO_ACTIVATION]:
            bucket[activated_name] = bucket.get(activated_name, 0) + 1
    return reports, confusion


def run_triggers(
    probes: ProbesFile,
    skills: list[SkillMeta],
    *,
    mode: Mode | None = None,
    llm: JsonCaller | None = None,
    runner: AgentRunner | None = None,
    limits: RunLimits | None = None,
) -> tuple[list[TriggerReport], dict[str, dict[str, int]]]:
    resolved: Mode = mode or probes.defaults.mode
    known = {s.name for s in skills}
    unknown = sorted({spec.skill for spec in probes.probes} - known)
    if unknown:
        raise ValueError(f"probes reference unknown skills: {', '.join(unknown)}")

    if resolved == "classifier" and llm is None:
        raise RuntimeError("classifier mode needs an LLM caller (set ANTHROPIC_API_KEY)")
    if resolved == "headless" and runner is None:
        raise RuntimeError("headless mode needs an agent runner")

    run_limits = limits or RunLimits(max_turns=2, max_budget_usd=0.05)
    results: list[TriggerResult] = []
    for text, intended in _probe_plan(probes):
        if resolved == "classifier":
            assert llm is not None
            activated = _classify(text, skills, llm, probes.defaults.classifier_model)
        else:
            assert runner is not None
            activated = _headless(text, skills, runner, run_limits)
        results.append(
            TriggerResult(
                probe=text,
                intended_skill=intended,
                expected=intended is not None,
                activated_skills=activated,
                mode=resolved,
            )
        )
    return score_results(probes, results, skills)


def cross_activation_pairs(
    confusion: dict[str, dict[str, int]],
    threshold: float = config.CROSS_ACTIVATION_MERGE_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Skill pairs that fire on each other's probes often enough to suggest a merge."""
    pairs: list[tuple[str, str, float]] = []
    for intended, activations in confusion.items():
        if intended == NEGATIVE_ROW:
            continue
        total = sum(activations.values()) or 1
        for activated, count in activations.items():
            if activated in {intended, NO_ACTIVATION}:
                continue
            fraction = count / total
            if fraction >= threshold:
                pairs.append((intended, activated, round(fraction, 3)))
    return sorted(pairs)


def dump_probe_template(skills: list[SkillMeta]) -> str:
    """Starter probes.yaml for a library that has none yet."""
    body = {
        "defaults": {"mode": "classifier", "classifier_model": config.DEFAULT_CLASSIFIER_MODEL},
        "shared_negatives": [
            "What's the capital of France?",
            "Write a bash script that rotates nginx logs.",
        ],
        "probes": [
            {
                "skill": s.name,
                "should_trigger": [f"TODO: a request that should load {s.name}"],
                "should_not_trigger": [f"TODO: a nearby request that must NOT load {s.name}"],
            }
            for s in skills
        ],
    }
    return yaml.safe_dump(json.loads(json.dumps(body)), sort_keys=False, width=100)
