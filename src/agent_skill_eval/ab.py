"""A/B differential engine: Control (--bare, no skill) vs Treatment (skill loaded)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent_skill_eval.llm import JsonCaller
from agent_skill_eval.metrics.stats import mean, median, pct_delta, rate
from agent_skill_eval.models.results import ABReport, ArmAggregate, TrialResult
from agent_skill_eval.models.task import LoadedTask
from agent_skill_eval.runner.base import AgentRunner
from agent_skill_eval.trial import Arm, run_trial

DEFAULT_TRIALS = 3


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def aggregate_arm(arm: Arm, trials: list[TrialResult]) -> ArmAggregate:
    totals = [float(t.metrics.total_tokens) for t in trials]
    return ArmAggregate(
        arm=arm,
        n=len(trials),
        pass_rate=rate(sum(t.passed for t in trials), len(trials)),
        mean_total_tokens=mean(totals),
        median_total_tokens=median(totals),
        mean_cost_usd=mean([t.metrics.cost_usd for t in trials]),
        mean_duration_seconds=mean([t.metrics.duration_seconds for t in trials]),
        mean_duplicate_calls=mean([float(t.metrics.duplicate_tool_calls) for t in trials]),
        mean_re_reads=mean([float(t.metrics.file_re_reads) for t in trials]),
        skill_activation_rate=rate(sum(t.skill_activated for t in trials), len(trials)),
        any_cost_estimated=any(t.metrics.cost_estimated for t in trials),
    )


def compute_deltas(control: ArmAggregate, treatment: ArmAggregate) -> dict[str, float]:
    """treatment - control; negative token/cost/duration deltas mean savings."""
    return {
        "pass_rate_delta": treatment.pass_rate - control.pass_rate,
        "token_delta_pct": pct_delta(treatment.mean_total_tokens, control.mean_total_tokens),
        "cost_delta_usd": treatment.mean_cost_usd - control.mean_cost_usd,
        "duration_delta_pct": pct_delta(
            treatment.mean_duration_seconds, control.mean_duration_seconds
        ),
        "duplicate_calls_delta": treatment.mean_duplicate_calls - control.mean_duplicate_calls,
        "re_reads_delta": treatment.mean_re_reads - control.mean_re_reads,
        "skill_activation_delta": (
            treatment.skill_activation_rate - control.skill_activation_rate
        ),
    }


def run_ab(
    loaded: LoadedTask,
    runners: dict[Arm, AgentRunner],
    *,
    trials: int | None = None,
    llm: JsonCaller | None = None,
    out_dir: Path | None = None,
    keep_workspace: bool = False,
) -> ABReport:
    n_trials = trials or loaded.spec.trials or DEFAULT_TRIALS
    all_trials: list[TrialResult] = []
    per_arm: dict[Arm, list[TrialResult]] = {"control": [], "treatment": []}
    arms: tuple[Arm, ...] = ("control", "treatment")
    for arm in arms:
        for index in range(n_trials):
            result = run_trial(
                loaded,
                runners[arm],
                arm,
                index,
                llm=llm,
                out_dir=out_dir,
                keep_workspace=keep_workspace,
            )
            per_arm[arm].append(result)
            all_trials.append(result)

    control = aggregate_arm("control", per_arm["control"])
    treatment = aggregate_arm("treatment", per_arm["treatment"])
    return ABReport(
        task_id=loaded.spec.id,
        skill_name=loaded.skill_name,
        control=control,
        treatment=treatment,
        deltas=compute_deltas(control, treatment),
        trials=all_trials,
        generated_at=utc_now(),
    )
