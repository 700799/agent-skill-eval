"""Composite scoring and KEEP / FIX / MERGE / RETIRE recommendations.

Each tier contributes a 0..1 component; missing tiers are dropped and the
remaining weights renormalized, so a library can be ranked from Tier 0 alone
and sharpened as evidence accumulates. Coverage flags travel with every entry
so a thin-evidence ranking never masquerades as a measured one.
"""

from __future__ import annotations

from datetime import UTC, datetime

from skill_eval_kit import config
from skill_eval_kit.metrics.stats import clamp, mean
from skill_eval_kit.models.portfolio import (
    CriticReport,
    LintReport,
    MiningReport,
    OverlapCluster,
    PortfolioReport,
    Recommendation,
    ScorecardEntry,
    SkillMeta,
    TriggerReport,
)
from skill_eval_kit.models.results import ABReport

#: Below this a skill with real evidence is not worth its context cost.
RETIRE_COMPOSITE = 0.35
#: Between RETIRE_COMPOSITE and this, a skill is salvageable but needs work.
FIX_COMPOSITE = 0.65
MIN_TRIGGER_F1 = 0.70
LOW_ADHERENCE = 0.50
#: An unmeasured skill is neither good nor bad; sparse evidence shrinks toward this.
NEUTRAL_PRIOR = 0.5


def ab_score(reports: list[ABReport]) -> tuple[float, dict[str, float]] | None:
    """Value-lift score from A/B evidence, plus the parts that produced it."""
    if not reports:
        return None
    pass_parts: list[float] = []
    token_parts: list[float] = []
    adherence_parts: list[float] = []
    for report in reports:
        pass_parts.append(clamp((report.deltas.get("pass_rate_delta", 0.0) + 1.0) / 2.0))
        control_tokens = report.control.mean_total_tokens
        saved = control_tokens - report.treatment.mean_total_tokens
        token_parts.append(clamp(0.5 + saved / max(control_tokens, 1.0)))
        adherence_parts.append(report.treatment.skill_activation_rate)
    parts = {
        "pass_lift": mean(pass_parts),
        "token_savings": mean(token_parts),
        "adherence": mean(adherence_parts),
    }
    score = 0.5 * parts["pass_lift"] + 0.25 * parts["token_savings"] + 0.25 * parts["adherence"]
    return score, parts


def composite_score(
    components: dict[str, float | None], weights: dict[str, float] | None = None
) -> float:
    active = weights or config.DEFAULT_WEIGHTS
    available = {k: v for k, v in components.items() if v is not None}
    total_weight = sum(active.get(k, 0.0) for k in available)
    if not available or total_weight <= 0:
        return 0.0
    return sum(active.get(k, 0.0) * v for k, v in available.items()) / total_weight


def coverage_confidence(
    components: dict[str, float | None], weights: dict[str, float] | None = None
) -> float:
    """Share of the total scoring weight that real evidence covers."""
    active = weights or config.DEFAULT_WEIGHTS
    total = sum(active.values()) or 1.0
    covered = sum(active.get(k, 0.0) for k, v in components.items() if v is not None)
    return covered / total


def shrink(composite: float, confidence: float) -> float:
    """Pull a composite toward the neutral prior in proportion to missing evidence.

    Without this, renormalizing over available tiers lets a skill with nothing
    but clean lint tie — or beat — a skill measured end-to-end and found good.
    """
    return composite * confidence + NEUTRAL_PRIOR * (1.0 - confidence)


def _cluster_for(skill: str, clusters: list[OverlapCluster]) -> OverlapCluster | None:
    for cluster in clusters:
        if skill in cluster.skills:
            return cluster
    return None


def _decide(
    entry: ScorecardEntry,
    *,
    lint: LintReport | None,
    ab_parts: dict[str, float] | None,
    cluster: OverlapCluster | None,
    cluster_best: str | None,
    cross_partners: list[str],
    mined_activations: int | None,
) -> tuple[Recommendation, list[str], list[str]]:
    reasons: list[str] = []
    tiers = sum(1 for present in entry.coverage.values() if present)
    evidence_beyond_lint = any(
        entry.coverage.get(tier) for tier in ("trigger", "ab", "critic")
    )

    # RETIRE — measurably not worth its cost.
    if entry.coverage.get("ab") and ab_parts is not None:
        no_lift = ab_parts["pass_lift"] <= 0.5
        no_savings = ab_parts["token_savings"] <= 0.5
        weak_trigger = entry.trigger_f1 is None or entry.trigger_f1 < 0.5
        if no_lift and no_savings and weak_trigger:
            reasons.append("A/B shows no pass-rate lift and no token savings")
            return "RETIRE", reasons, []
    if evidence_beyond_lint and tiers >= 2 and entry.composite < RETIRE_COMPOSITE:
        reasons.append(
            f"composite {entry.composite:.2f} below {RETIRE_COMPOSITE} with real evidence"
        )
        return "RETIRE", reasons, []
    if mined_activations == 0 and evidence_beyond_lint and entry.composite < 0.5:
        reasons.append("never activated in mined sessions and scores poorly")
        return "RETIRE", reasons, []

    # MERGE — redundant with a stronger sibling.
    if cluster and cluster_best and cluster_best != entry.skill:
        others = [s for s in cluster.skills if s != entry.skill]
        reasons.append(
            f"overlaps {', '.join(others)} (similarity {cluster.max_similarity:.2f}); "
            f"{cluster_best} scores higher"
        )
        return "MERGE", reasons, [cluster_best]
    if cross_partners:
        reasons.append(f"fires on probes intended for {', '.join(cross_partners)}")
        return "MERGE", reasons, cross_partners

    # FIX — salvageable, but not earning its keep as written.
    lint_errors = [f.check for f in (lint.findings if lint else []) if f.severity == "error"]
    if lint_errors:
        reasons.append(f"blocking lint findings: {', '.join(sorted(set(lint_errors)))}")
    if entry.trigger_f1 is not None and entry.trigger_f1 < MIN_TRIGGER_F1:
        reasons.append(f"trigger F1 {entry.trigger_f1:.2f} below {MIN_TRIGGER_F1}")
    if ab_parts is not None and ab_parts["adherence"] < LOW_ADHERENCE:
        # The classic bypass signature: tasks pass, but not because of the skill.
        reasons.append(
            f"adherence {ab_parts['adherence']:.2f}: tasks pass without the skill being used"
        )
    if RETIRE_COMPOSITE <= entry.composite < FIX_COMPOSITE:
        reasons.append(f"composite {entry.composite:.2f} is mid-range")
    if reasons:
        return "FIX", reasons, []

    if not evidence_beyond_lint:
        reasons.append("insufficient_evidence: only static lint has run")
    else:
        reasons.append("clears every tier that has run")
    return "KEEP", reasons, []


def build_scorecard(
    skills: list[SkillMeta],
    *,
    lint_reports: list[LintReport] | None = None,
    trigger_reports: list[TriggerReport] | None = None,
    ab_reports: list[ABReport] | None = None,
    critic_reports: list[CriticReport] | None = None,
    mining: MiningReport | None = None,
    clusters: list[OverlapCluster] | None = None,
    cross_activation: list[tuple[str, str, float]] | None = None,
    weights: dict[str, float] | None = None,
) -> PortfolioReport:
    lint_by = {r.skill: r for r in (lint_reports or [])}
    trigger_by = {r.skill: r for r in (trigger_reports or [])}
    critic_by = {r.skill: r for r in (critic_reports or [])}
    mined_by = {u.skill: u for u in (mining.skills if mining else [])}
    ab_by: dict[str, list[ABReport]] = {}
    for report in ab_reports or []:
        if report.skill_name:
            ab_by.setdefault(report.skill_name, []).append(report)
    all_clusters = clusters or []

    entries: list[ScorecardEntry] = []
    scored: dict[str, float] = {}
    parts_by: dict[str, dict[str, float] | None] = {}

    # First pass: components and composites (cluster ranking needs them).
    drafts: list[tuple[SkillMeta, ScorecardEntry]] = []
    for meta in skills:
        lint = lint_by.get(meta.name)
        trigger = trigger_by.get(meta.name)
        critic = critic_by.get(meta.name)
        ab_outcome = ab_score(ab_by.get(meta.name, []))
        ab_value = ab_outcome[0] if ab_outcome else None
        parts_by[meta.name] = ab_outcome[1] if ab_outcome else None

        components: dict[str, float | None] = {
            "lint": lint.score if lint else None,
            "trigger": trigger.f1 if trigger else None,
            "ab": ab_value,
            "critic": critic.score if critic else None,
        }
        composite = composite_score(components, weights)
        confidence = coverage_confidence(components, weights)
        ranked = shrink(composite, confidence)
        scored[meta.name] = ranked
        mined = mined_by.get(meta.name)
        entry = ScorecardEntry(
            skill=meta.name,
            lint_score=components["lint"],
            trigger_f1=components["trigger"],
            ab_score=components["ab"],
            critic_score=components["critic"],
            mined_activations=mined.activations if mined else None,
            coverage={
                "lint": lint is not None,
                "trigger": trigger is not None,
                "ab": ab_value is not None,
                "critic": critic is not None,
            },
            composite=round(composite, 4),
            confidence=round(confidence, 4),
            ranked_score=round(ranked, 4),
        )
        drafts.append((meta, entry))

    cross_by: dict[str, list[str]] = {}
    for intended, activated, _fraction in cross_activation or []:
        if scored.get(activated, 0.0) > scored.get(intended, 0.0):
            cross_by.setdefault(intended, []).append(activated)

    for meta, entry in drafts:
        cluster = _cluster_for(meta.name, all_clusters)
        # Survivor is the best-evidenced member, not merely the best-scoring one.
        cluster_best = (
            max(cluster.skills, key=lambda s: scored.get(s, 0.0)) if cluster else None
        )
        recommendation, reasons, merge_with = _decide(
            entry,
            lint=lint_by.get(meta.name),
            ab_parts=parts_by.get(meta.name),
            cluster=cluster,
            cluster_best=cluster_best,
            cross_partners=sorted(set(cross_by.get(meta.name, []))),
            mined_activations=entry.mined_activations,
        )
        entry.recommendation = recommendation
        entry.reasons = reasons
        entry.merge_with = merge_with
        entries.append(entry)

    for cluster in all_clusters:
        cluster.merge_into = max(cluster.skills, key=lambda s: scored.get(s, 0.0))

    entries.sort(key=lambda e: (-e.ranked_score, e.skill))
    return PortfolioReport(
        entries=entries,
        clusters=all_clusters,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
