"""Static quality gate: ruff and mypy over the final workspace state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent_skill_eval.evaluators.base import EvalContext, register
from agent_skill_eval.models.results import EvalResult
from agent_skill_eval.models.task import PydevStaticEval

_SUBPROCESS_TIMEOUT = 120
#: mypy runs with missing third-party stubs tolerated: eval workspaces have no
#: venv of their own, so import resolution errors would swamp real findings.
_MYPY_BASE = ["--no-error-summary", "--ignore-missing-imports"]


def _run_tool(args: list[str], cwd: Path) -> tuple[int, str, str] | None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.returncode, proc.stdout, proc.stderr


def _ruff_issues(workspace: Path, paths: list[str], strict: bool) -> tuple[int, list[str]] | None:
    # --no-cache keeps the graded workspace free of .ruff_cache artifacts.
    args = ["ruff", "check", "--no-cache", "--output-format", "json", "--exclude", ".claude"]
    if strict:
        args += ["--select", "E,W,F,I,N,UP,B,SIM,C4"]
    result = _run_tool(args + paths, workspace)
    if result is None:
        return None
    _, stdout, _ = result
    try:
        diagnostics = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return None
    messages = [
        f"{d.get('filename', '?')}:{d.get('location', {}).get('row', '?')} "
        f"{d.get('code', '?')} {d.get('message', '')}"
        for d in diagnostics
        if isinstance(d, dict)
    ]
    return len(messages), messages


def _mypy_issues(workspace: Path, paths: list[str], strict: bool) -> tuple[int, list[str]] | None:
    args = ["mypy", *_MYPY_BASE]
    if strict:
        args.append("--strict")
    result = _run_tool(args + paths + ["--exclude", r"\.claude"], workspace)
    if result is None:
        return None
    _, stdout, _ = result
    messages = [line for line in stdout.splitlines() if ": error:" in line]
    return len(messages), messages


@register("pydev_static")
def evaluate(spec: PydevStaticEval, ctx: EvalContext) -> EvalResult:
    details: dict[str, object] = {}
    total_issues = 0
    failures: list[str] = []
    for linter in spec.linters:
        runner = _ruff_issues if linter == "ruff" else _mypy_issues
        outcome = runner(ctx.workspace, spec.paths, spec.strict)
        if outcome is None:
            failures.append(linter)
            details[linter] = {"error": "tool unavailable or failed to run"}
            continue
        count, messages = outcome
        total_issues += count
        details[linter] = {"issues": count, "messages": messages[:10]}

    if failures:
        return EvalResult(
            name="pydev_static",
            passed=False,
            score=0.0,
            required=spec.required,
            details=details,
        )
    return EvalResult(
        name="pydev_static",
        passed=total_issues == 0,
        score=1.0 if total_issues == 0 else max(0.0, 1.0 - 0.1 * total_issues),
        required=spec.required,
        details=details,
    )
