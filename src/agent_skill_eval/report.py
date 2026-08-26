"""Markdown rendering for task runs, A/B differentials, and the leaderboard."""

from __future__ import annotations

from typing import Any

from agent_skill_eval.models.portfolio import PortfolioReport
from agent_skill_eval.models.results import ABReport, EvalResult, TaskReport, TrialResult


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.0f}pp"


def _signed_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _eval_note(result: EvalResult) -> str:
    details = result.details
    for key in ("skipped", "error", "reasoning"):
        if key in details:
            return str(details[key])[:100]
    if result.name == "skill_activation":
        return f"activations={details.get('activations')}"
    if result.name == "trajectory_efficiency":
        return (
            f"dups={details.get('duplicate_tool_calls')} "
            f"re-reads={details.get('file_re_reads')} "
            f"max-growth={details.get('max_context_growth_tokens')}"
        )
    if result.name == "pydev_static":
        return " ".join(
            f"{linter}={info.get('issues', '?') if isinstance(info, dict) else '?'}"
            for linter, info in details.items()
        )
    if result.name == "pydev_tests":
        return f"passed={details.get('passed')} failed={details.get('failed')}"
    if result.name == "ast_assertions":
        assertions = details.get("assertions", [])
        failing = [a for a in assertions if isinstance(a, dict) and not a.get("ok")]
        return f"{len(assertions) - len(failing)}/{len(assertions)} assertions ok"
    return ""


def _trial_section(trial: TrialResult) -> list[str]:
    flag = "PASS" if trial.passed else "FAIL"
    metrics = trial.metrics
    est = " (est.)" if metrics.cost_estimated else ""
    lines = [
        f"### {trial.arm} · trial {trial.trial_index} — {flag} ({trial.outcome_subtype})",
        "",
        f"- tokens: {metrics.total_tokens:,} · cost: ${metrics.cost_usd:.4f}{est} · "
        f"turns: {metrics.num_turns} · duration: {metrics.duration_seconds:.1f}s",
        f"- waste: {metrics.duplicate_tool_calls} duplicate calls, "
        f"{metrics.file_re_reads} re-reads, "
        f"max context growth {metrics.max_context_growth:,} tokens"
        + (" (est.)" if metrics.context_growth_estimated else ""),
        "",
        "| evaluator | passed | score | notes |",
        "|---|---|---|---|",
    ]
    for result in trial.eval_results:
        mark = "✅" if result.passed else ("⏭️" if "skipped" in result.details else "❌")
        lines.append(f"| {result.name} | {mark} | {result.score:.2f} | {_eval_note(result)} |")
    lines.append("")
    return lines


def render_task_report(report: TaskReport) -> str:
    lines = [
        f"# Run report: `{report.task_id}`",
        "",
        f"Arm: **{report.arm}** · Skill: `{report.skill_name or '—'}` · "
        f"Generated: {report.generated_at}",
        "",
    ]
    for trial in report.trials:
        lines.extend(_trial_section(trial))
    return "\n".join(lines)


def render_ab_report(report: ABReport) -> str:
    control, treatment, deltas = report.control, report.treatment, report.deltas
    estimated = control.any_cost_estimated or treatment.any_cost_estimated
    est = " (some costs estimated)" if estimated else ""
    lines = [
        f"# A/B report: `{report.task_id}`",
        "",
        f"Skill: `{report.skill_name or '—'}` · {control.n}+{treatment.n} trials · "
        f"Generated: {report.generated_at}{est}",
        "",
        "Deltas are treatment − control: negative token/cost/duration deltas mean "
        "the skill saves; positive pass-rate delta means the skill helps.",
        "",
        "| metric | control | treatment | delta |",
        "|---|---|---|---|",
        f"| pass rate | {_pct(control.pass_rate)} | {_pct(treatment.pass_rate)} | "
        f"{_pp(deltas.get('pass_rate_delta', 0.0))} |",
        f"| mean total tokens | {control.mean_total_tokens:,.0f} | "
        f"{treatment.mean_total_tokens:,.0f} | {_signed_pct(deltas.get('token_delta_pct', 0.0))} |",
        f"| mean cost (USD) | ${control.mean_cost_usd:.4f} | ${treatment.mean_cost_usd:.4f} | "
        f"{deltas.get('cost_delta_usd', 0.0):+.4f} |",
        f"| mean duration (s) | {control.mean_duration_seconds:.1f} | "
        f"{treatment.mean_duration_seconds:.1f} | "
        f"{_signed_pct(deltas.get('duration_delta_pct', 0.0))} |",
        f"| duplicate tool calls | {control.mean_duplicate_calls:.1f} | "
        f"{treatment.mean_duplicate_calls:.1f} | {deltas.get('duplicate_calls_delta', 0.0):+.1f} |",
        f"| file re-reads | {control.mean_re_reads:.1f} | {treatment.mean_re_reads:.1f} | "
        f"{deltas.get('re_reads_delta', 0.0):+.1f} |",
        f"| skill activation | {_pct(control.skill_activation_rate)} | "
        f"{_pct(treatment.skill_activation_rate)} | "
        f"{_pp(deltas.get('skill_activation_delta', 0.0))} |",
        "",
        "## Trials",
        "",
    ]
    for trial in report.trials:
        lines.extend(_trial_section(trial))
    return "\n".join(lines)


def _fmt_component(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "—"


def render_leaderboard(report: PortfolioReport) -> str:
    lines = [
        "# Skill portfolio leaderboard",
        "",
        f"Generated: {report.generated_at}",
        "",
        "| rank | skill | score | confidence | recommendation | lint | trigger F1 | A/B | "
        "critic | mined uses | reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, entry in enumerate(report.entries, start=1):
        recommendation: str = entry.recommendation
        if entry.merge_with:
            recommendation += f" → {', '.join(entry.merge_with)}"
        mined = str(entry.mined_activations) if entry.mined_activations is not None else "—"
        lines.append(
            f"| {rank} | `{entry.skill}` | {entry.composite:.2f} | "
            f"{entry.confidence * 100:.0f}% | **{recommendation}** | "
            f"{_fmt_component(entry.lint_score)} | {_fmt_component(entry.trigger_f1)} | "
            f"{_fmt_component(entry.ab_score)} | {_fmt_component(entry.critic_score)} | "
            f"{mined} | {'; '.join(entry.reasons)[:160]} |"
        )
    lines.append("")
    lines.append(
        "Missing tiers show as `—`. Confidence is the share of scoring weight backed by "
        "evidence; ranking shrinks low-confidence scores toward neutral so an unmeasured "
        "skill cannot outrank a measured one, and thin coverage is never retired."
    )
    if report.clusters:
        lines += ["", "## Overlap clusters", ""]
        for cluster in report.clusters:
            merge = f" → merge into `{cluster.merge_into}`" if cluster.merge_into else ""
            lines.append(
                f"- {', '.join(f'`{s}`' for s in cluster.skills)} "
                f"(max similarity {cluster.max_similarity:.2f}){merge} {cluster.rationale}"
            )
    if report.confusion:
        lines += ["", "## Trigger confusion matrix (rows: intended, columns: activated)", ""]
        columns = sorted({col for row in report.confusion.values() for col in row})
        lines.append("| intended \\ activated | " + " | ".join(columns) + " |")
        lines.append("|---|" + "---|" * len(columns))
        for row_name in sorted(report.confusion):
            row: dict[str, Any] = report.confusion[row_name]
            cells = " | ".join(str(row.get(col, 0)) for col in columns)
            lines.append(f"| {row_name} | {cells} |")
    lines.append("")
    return "\n".join(lines)
