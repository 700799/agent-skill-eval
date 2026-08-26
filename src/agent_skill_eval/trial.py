"""Run one trial: workspace -> runner -> metrics -> evaluators -> TrialResult."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent_skill_eval.evaluators import EvalContext, run_evaluations
from agent_skill_eval.llm import JsonCaller
from agent_skill_eval.metrics import loops, tokens
from agent_skill_eval.models.results import TrialMetrics, TrialResult
from agent_skill_eval.models.task import LoadedTask, TrajectoryEfficiencyEval
from agent_skill_eval.models.trajectory import Trajectory
from agent_skill_eval.runner.base import AgentRunner
from agent_skill_eval.runner.workspace import materialize, sandbox

Arm = Literal["control", "treatment"]


def _efficiency_spec(loaded: LoadedTask) -> TrajectoryEfficiencyEval:
    for spec in loaded.spec.evaluations:
        if isinstance(spec, TrajectoryEfficiencyEval):
            return spec
    return TrajectoryEfficiencyEval(type="trajectory_efficiency")


def compute_metrics(trajectory: Trajectory, loaded: LoadedTask) -> TrialMetrics:
    usage, usage_estimated = tokens.trajectory_usage(trajectory)
    cost, cost_estimated = tokens.trajectory_cost(trajectory)
    thresholds = _efficiency_spec(loaded)
    loop_metrics = loops.analyze(
        trajectory,
        exempt_tools=thresholds.exempt_tools,
        context_threshold_tokens=thresholds.max_context_growth_tokens,
    )
    return TrialMetrics(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        cache_creation_tokens=usage.cache_creation_input_tokens,
        cost_usd=cost,
        cost_estimated=cost_estimated or usage_estimated,
        duration_seconds=trajectory.outcome.duration_seconds or 0.0,
        num_turns=trajectory.outcome.num_turns or len(trajectory.turns),
        duplicate_tool_calls=loop_metrics.duplicates.duplicate_tool_calls,
        file_re_reads=loop_metrics.re_reads.file_re_reads,
        max_context_growth=loop_metrics.context_growth.max_growth,
        context_growth_estimated=loop_metrics.context_growth.estimated,
    )


def run_trial(
    loaded: LoadedTask,
    runner: AgentRunner,
    arm: Arm,
    trial_index: int,
    *,
    llm: JsonCaller | None = None,
    out_dir: Path | None = None,
    keep_workspace: bool = False,
) -> TrialResult:
    trial_dir = out_dir / arm / f"trial_{trial_index}" if out_dir else None
    raw_out = trial_dir / "raw.ndjson" if trial_dir else None
    skill_path = loaded.skill_path if arm == "treatment" else None

    def execute(workspace: Path) -> TrialResult:
        trajectory = runner.run(
            loaded.spec.prompt,
            workspace,
            loaded.spec.run_limits,
            no_skills=arm == "control",
            raw_out=raw_out,
        )
        metrics = compute_metrics(trajectory, loaded)
        eval_results = run_evaluations(
            loaded.spec,
            EvalContext(
                task=loaded.spec,
                trajectory=trajectory,
                workspace=workspace,
                skill_name=loaded.skill_name,
                llm=llm,
            ),
        )
        passed = trajectory.outcome.is_success and all(
            r.passed for r in eval_results if r.required
        )
        skill_activated = bool(
            loaded.skill_name and loaded.skill_name in trajectory.skill_activations()
        )
        return TrialResult(
            task_id=loaded.spec.id,
            arm=arm,
            trial_index=trial_index,
            outcome_subtype=trajectory.outcome.subtype,
            skill_activated=skill_activated,
            metrics=metrics,
            eval_results=eval_results,
            passed=passed,
            ndjson_path=str(raw_out) if raw_out else None,
        )

    if keep_workspace and trial_dir is not None:
        workspace = trial_dir / "workspace"
        materialize(loaded.spec.workspace, workspace)
        if skill_path is not None:
            from agent_skill_eval.runner.workspace import inject_skill

            inject_skill(workspace, skill_path)
        return execute(workspace)
    with sandbox(loaded.spec.workspace, skill_path=skill_path) as workspace:
        return execute(workspace)
