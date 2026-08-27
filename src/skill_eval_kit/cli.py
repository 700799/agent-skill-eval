"""`skill-eval` — command-line entry point for skill-eval-kit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from skill_eval_kit import __version__, config
from skill_eval_kit.ab import DEFAULT_TRIALS, run_ab, utc_now
from skill_eval_kit.llm import JsonCaller, default_caller
from skill_eval_kit.models.portfolio import (
    AuditReport,
    CriticReport,
    MiningReport,
    TriggerReport,
)
from skill_eval_kit.models.results import ABReport, TaskReport
from skill_eval_kit.models.task import RunLimits, load_task
from skill_eval_kit.portfolio import (
    build_scorecard,
    cluster_overlaps,
    critique_all,
    discover_skills,
    lint_all,
    load_probes,
    mine_sessions,
    run_triggers,
)
from skill_eval_kit.portfolio.skills import parse_skill
from skill_eval_kit.portfolio.triggers import cross_activation_pairs
from skill_eval_kit.report import render_ab_report, render_leaderboard, render_task_report
from skill_eval_kit.runner.base import AgentRunner
from skill_eval_kit.runner.claude_cli import ClaudeCliRunner
from skill_eval_kit.runner.replay import ReplayRunner
from skill_eval_kit.runs import new_run_dir, save_json, save_meta
from skill_eval_kit.trial import run_trial

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _make_runner(
    kind: str, fixtures: list[str] | None, claude_bin: str
) -> AgentRunner:
    if kind == "replay":
        if not fixtures:
            raise SystemExit("--runner replay needs at least one --fixture")
        return ReplayRunner([Path(f) for f in fixtures])
    return ClaudeCliRunner(claude_bin=claude_bin)


def _resolve_llm(no_llm: bool) -> JsonCaller | None:
    return None if no_llm else default_caller()


def _task_paths(target: str) -> list[Path]:
    path = Path(target)
    if path.is_dir():
        return sorted(p for p in path.glob("*.y*ml"))
    return [path]


# --------------------------------------------------------------------------- validate


def cmd_validate(args: argparse.Namespace) -> int:
    errors = 0
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"MISSING  {path}")
            errors += 1
            continue
        kind = args.kind
        if kind == "auto":
            name = path.name.lower()
            if "probe" in name:
                kind = "probes"
            elif path.suffix.lower() == ".md":
                kind = "skill"
            else:
                kind = "task"
        try:
            if kind == "task":
                loaded = load_task(path)
                evals = ", ".join(e.type for e in loaded.spec.evaluations)
                print(f"OK       {path}  [task {loaded.spec.id}] evaluations: {evals}")
            elif kind == "probes":
                probes = load_probes(path)
                total = sum(len(p.should_trigger) for p in probes.probes)
                print(
                    f"OK       {path}  [probes] {len(probes.probes)} skills, "
                    f"{total} positive probes, {len(probes.shared_negatives)} shared negatives"
                )
            else:
                meta = parse_skill(path)
                print(f"OK       {path}  [skill {meta.name}] ~{meta.token_estimate} tokens")
        except Exception as exc:  # noqa: BLE001 - report, don't traceback at the CLI
            print(f"INVALID  {path}\n         {exc}")
            errors += 1
    return EXIT_FAILED if errors else EXIT_OK


# --------------------------------------------------------------------------- run / ab


def _apply_overrides(limits: RunLimits, args: argparse.Namespace) -> RunLimits:
    updates: dict[str, Any] = {}
    if getattr(args, "model", None):
        updates["model"] = args.model
    if getattr(args, "max_turns", None):
        updates["max_turns"] = args.max_turns
    if getattr(args, "max_budget_usd", None):
        updates["max_budget_usd"] = args.max_budget_usd
    return limits.model_copy(update=updates) if updates else limits


def cmd_run(args: argparse.Namespace) -> int:
    loaded = load_task(args.task)
    loaded.spec.run_limits = _apply_overrides(loaded.spec.run_limits, args)
    runner = _make_runner(args.runner, args.fixture, args.claude_bin)
    out_dir = new_run_dir(Path(args.out_dir), loaded.spec.id)
    save_meta(out_dir, {"command": "run", "arm": args.arm, "runner": args.runner})

    trials = [
        run_trial(
            loaded,
            runner,
            args.arm,
            index,
            llm=_resolve_llm(args.no_llm),
            out_dir=out_dir,
            keep_workspace=args.keep_workspace,
        )
        for index in range(args.trials)
    ]
    report = TaskReport(
        task_id=loaded.spec.id,
        skill_name=loaded.skill_name,
        arm=args.arm,
        trials=trials,
        generated_at=utc_now(),
    )
    save_json(report, out_dir / "run_report.json")
    markdown = render_task_report(report)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nArtifacts: {out_dir}")
    return EXIT_OK if all(t.passed for t in trials) else EXIT_FAILED


def cmd_ab(args: argparse.Namespace) -> int:
    llm = _resolve_llm(args.no_llm)
    overall_ok = True
    for task_path in _task_paths(args.task):
        loaded = load_task(task_path)
        loaded.spec.run_limits = _apply_overrides(loaded.spec.run_limits, args)
        runners: dict[str, AgentRunner] = {
            "control": _make_runner(args.runner, args.control_fixture, args.claude_bin),
            "treatment": _make_runner(args.runner, args.treatment_fixture, args.claude_bin),
        }
        out_dir = new_run_dir(Path(args.out_dir), loaded.spec.id)
        save_meta(out_dir, {"command": "ab", "runner": args.runner, "trials": args.trials})
        report = run_ab(
            loaded,
            runners,  # type: ignore[arg-type]
            trials=args.trials,
            llm=llm,
            out_dir=out_dir,
            keep_workspace=args.keep_workspace,
        )
        save_json(report, out_dir / "ab_report.json")
        markdown = render_ab_report(report)
        (out_dir / "report.md").write_text(markdown, encoding="utf-8")
        if args.json:
            Path(args.json).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if args.md:
            Path(args.md).write_text(markdown, encoding="utf-8")
        print(markdown)
        print(f"\nArtifacts: {out_dir}")
        overall_ok &= report.deltas.get("pass_rate_delta", 0.0) >= 0
    return EXIT_OK if overall_ok else EXIT_FAILED


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    ab_path, run_path = run_dir / "ab_report.json", run_dir / "run_report.json"
    if ab_path.is_file():
        report: ABReport | TaskReport = ABReport.model_validate_json(
            ab_path.read_text(encoding="utf-8")
        )
        markdown = render_ab_report(report)  # type: ignore[arg-type]
    elif run_path.is_file():
        report = TaskReport.model_validate_json(run_path.read_text(encoding="utf-8"))
        markdown = render_task_report(report)
    else:
        print(f"no ab_report.json or run_report.json under {run_dir}", file=sys.stderr)
        return EXIT_USAGE
    print(report.model_dump_json(indent=2) if args.format == "json" else markdown)
    return EXIT_OK


# --------------------------------------------------------------------------- portfolio


def cmd_audit(args: argparse.Namespace) -> int:
    skills = discover_skills(args.skills_dir)
    if not skills:
        print(f"no skills found under {args.skills_dir}", file=sys.stderr)
        return EXIT_USAGE
    lint_reports = lint_all(skills)
    clusters = cluster_overlaps(skills, threshold=args.similarity_threshold)

    trigger_reports: list[TriggerReport] = []
    confusion: dict[str, dict[str, int]] | None = None
    if args.probes:
        probes = load_probes(args.probes)
        mode = args.trigger_mode or probes.defaults.mode
        runner = (
            ClaudeCliRunner(claude_bin=args.claude_bin) if mode == "headless" else None
        )
        trigger_reports, confusion = run_triggers(
            probes,
            skills,
            mode=mode,
            llm=_resolve_llm(args.no_llm),
            runner=runner,
        )

    critics: list[CriticReport] = []
    if args.critic:
        llm = _resolve_llm(args.no_llm)
        if llm is None:
            print("--critic needs API credentials (or drop --critic)", file=sys.stderr)
            return EXIT_USAGE
        critics = critique_all(
            skills, llm, model=args.critic_model, lint_by_skill={r.skill: r for r in lint_reports}
        )

    audit = AuditReport(
        skills_dir=str(args.skills_dir),
        lint=lint_reports,
        clusters=clusters,
        triggers=trigger_reports,
        confusion=confusion,
        critics=critics,
        generated_at=utc_now(),
    )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print(f"# Audit: {args.skills_dir}  ({len(skills)} skills)\n")
    for lint_report in sorted(lint_reports, key=lambda r: r.score):
        print(f"{lint_report.score:.2f}  {lint_report.skill}")
        for finding in lint_report.findings:
            print(f"        [{finding.severity}] {finding.check}: {finding.message}")
    if clusters:
        print("\n## Overlap clusters")
        for cluster in clusters:
            print(f"  {', '.join(cluster.skills)}  (similarity {cluster.max_similarity:.2f})")
    if trigger_reports:
        print("\n## Trigger accuracy")
        for trigger_report in trigger_reports:
            print(
                f"  {trigger_report.skill}: F1 {trigger_report.f1:.2f} "
                f"(precision {trigger_report.precision:.2f}, "
                f"recall {trigger_report.recall:.2f}, tp={trigger_report.tp} "
                f"fp={trigger_report.fp} fn={trigger_report.fn})"
            )
    if critics:
        print("\n## Critic")
        for critic in critics:
            print(f"  {critic.score:.2f}  {critic.skill}")
            for issue in critic.issues[:5]:
                print(f"        - {issue}")
    if args.json:
        print(f"\nWrote {args.json}")
    return EXIT_OK


def cmd_mine(args: argparse.Namespace) -> int:
    known = None
    if args.skills_dir:
        known = [s.name for s in discover_skills(args.skills_dir)]
    report = mine_sessions(list(args.paths), known_skills=known)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"# Mined {report.sessions_scanned} session transcript(s)\n")
    print(f"{'skill':<32} {'uses':>6} {'sessions':>9} {'est tokens':>12} {'est cost':>10}")
    for usage in report.skills:
        print(
            f"{usage.skill:<32} {usage.activations:>6} {usage.sessions:>9} "
            f"{usage.est_tokens:>12,} {usage.est_cost_usd:>10.4f}"
        )
    if args.json:
        print(f"\nWrote {args.json}")
    return EXIT_OK


def _parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    weights = dict(config.DEFAULT_WEIGHTS)
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        key = key.strip()
        if key not in weights:
            raise SystemExit(f"unknown weight {key!r}; expected one of {sorted(weights)}")
        weights[key] = float(value)
    return weights


def cmd_rank(args: argparse.Namespace) -> int:
    skills = discover_skills(args.skills_dir)
    lint_reports = lint_all(skills)
    clusters = cluster_overlaps(skills, threshold=args.similarity_threshold)
    trigger_reports: list[TriggerReport] = []
    critics: list[CriticReport] = []
    cross: list[tuple[str, str, float]] = []

    if args.audit:
        audit = AuditReport.model_validate_json(Path(args.audit).read_text(encoding="utf-8"))
        lint_reports = audit.lint or lint_reports
        clusters = audit.clusters or clusters
        trigger_reports = audit.triggers
        critics = audit.critics
        if audit.confusion:
            cross = cross_activation_pairs(audit.confusion)

    ab_reports: list[ABReport] = []
    if args.runs:
        from skill_eval_kit.runs import load_ab_reports

        ab_reports = load_ab_reports(Path(args.runs))

    mining = None
    if args.mine:
        mining = MiningReport.model_validate_json(Path(args.mine).read_text(encoding="utf-8"))

    report = build_scorecard(
        skills,
        lint_reports=lint_reports,
        trigger_reports=trigger_reports,
        ab_reports=ab_reports,
        critic_reports=critics,
        mining=mining,
        clusters=clusters,
        cross_activation=cross,
        weights=_parse_weights(args.weights),
    )
    if args.audit and trigger_reports:
        audit_confusion = AuditReport.model_validate_json(
            Path(args.audit).read_text(encoding="utf-8")
        ).confusion
        report.confusion = audit_confusion

    markdown = render_leaderboard(report)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(markdown, encoding="utf-8")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(markdown)
    if args.md:
        print(f"Wrote {args.md}")
    return EXIT_OK


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skill-eval",
        description="Evaluate Claude Code skills: A/B differentials, token waste, "
        "skill adherence, and whole-library ranking.",
    )
    parser.add_argument("--version", action="version", version=f"skill-eval-kit {__version__}")
    parser.add_argument(
        "--out-dir", default="runs", help="where run artifacts land (default: runs)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="never call the API: skips llm_judge, critic, and classifier triggers",
    )
    parser.add_argument("--claude-bin", default="claude", help="claude CLI binary name/path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="schema-check task/probes/skill files")
    p_validate.add_argument("paths", nargs="+")
    p_validate.add_argument("--kind", choices=["auto", "task", "probes", "skill"], default="auto")
    p_validate.set_defaults(func=cmd_validate)

    def add_run_limits(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="override run_limits.model")
        p.add_argument("--max-turns", type=int, help="override run_limits.max_turns")
        p.add_argument("--max-budget-usd", type=float, help="override run_limits.max_budget_usd")
        p.add_argument("--runner", choices=["claude", "replay"], default="claude")
        p.add_argument("--keep-workspace", action="store_true")

    p_run = sub.add_parser("run", help="run one arm of a task (debugging)")
    p_run.add_argument("task")
    p_run.add_argument("--arm", choices=["control", "treatment"], default="treatment")
    p_run.add_argument("--trials", type=int, default=1)
    p_run.add_argument("--fixture", action="append", help="replay NDJSON (repeatable)")
    add_run_limits(p_run)
    p_run.set_defaults(func=cmd_run)

    p_ab = sub.add_parser("ab", help="A/B a task: control (no skill) vs treatment")
    p_ab.add_argument("task", help="task.yaml or a directory of them")
    p_ab.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p_ab.add_argument("--control-fixture", action="append")
    p_ab.add_argument("--treatment-fixture", action="append")
    p_ab.add_argument("--json", help="also write the ABReport JSON here")
    p_ab.add_argument("--md", help="also write the Markdown report here")
    add_run_limits(p_ab)
    p_ab.set_defaults(func=cmd_ab)

    p_report = sub.add_parser("report", help="re-render a stored run directory")
    p_report.add_argument("run_dir")
    p_report.add_argument("--format", choices=["md", "json"], default="md")
    p_report.set_defaults(func=cmd_report)

    p_audit = sub.add_parser("audit", help="Tier 0/1 audit of a skill library")
    p_audit.add_argument("skills_dir")
    p_audit.add_argument("--probes", help="probes.yaml enables trigger accuracy (Tier 1)")
    p_audit.add_argument("--trigger-mode", choices=["classifier", "headless"])
    p_audit.add_argument("--critic", action="store_true", help="add LLM critique (costs tokens)")
    p_audit.add_argument("--critic-model", default=config.DEFAULT_CRITIC_MODEL)
    p_audit.add_argument(
        "--similarity-threshold", type=float, default=config.DEFAULT_SIMILARITY_THRESHOLD
    )
    p_audit.add_argument("--json", help="write the AuditReport JSON here")
    p_audit.set_defaults(func=cmd_audit)

    p_mine = sub.add_parser("mine", help="mine real session transcripts for skill usage")
    p_mine.add_argument("paths", nargs="+")
    p_mine.add_argument("--skills-dir", help="restrict to skills found in this library")
    p_mine.add_argument("--json")
    p_mine.set_defaults(func=cmd_mine)

    p_rank = sub.add_parser("rank", help="compose stored artifacts into a leaderboard")
    p_rank.add_argument("skills_dir")
    p_rank.add_argument("--audit", help="audit.json from `ase audit`")
    p_rank.add_argument("--runs", help="runs/ directory holding ab_report.json files")
    p_rank.add_argument("--mine", help="mine.json from `ase mine`")
    p_rank.add_argument("--weights", help="e.g. lint=0.1,trigger=0.2,ab=0.5,critic=0.2")
    p_rank.add_argument(
        "--similarity-threshold", type=float, default=config.DEFAULT_SIMILARITY_THRESHOLD
    )
    p_rank.add_argument("--md")
    p_rank.add_argument("--json")
    p_rank.set_defaults(func=cmd_rank)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result: int = args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return result


if __name__ == "__main__":
    raise SystemExit(main())
