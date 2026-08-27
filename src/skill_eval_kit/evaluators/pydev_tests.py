"""Functional gate: run pytest inside the final workspace."""

from __future__ import annotations

import re
import subprocess
import sys

from skill_eval_kit.evaluators.base import EvalContext, register
from skill_eval_kit.models.results import EvalResult
from skill_eval_kit.models.task import PydevTestsEval

_SUBPROCESS_TIMEOUT = 300


@register("pydev_tests")
def evaluate(spec: PydevTestsEval, ctx: EvalContext) -> EvalResult:
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--ignore",
        ".claude",
        *spec.test_paths,
    ]
    try:
        proc = subprocess.run(
            argv, cwd=ctx.workspace, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EvalResult(
            name="pydev_tests",
            passed=False,
            score=0.0,
            required=spec.required,
            details={"error": f"pytest could not run: {exc}"},
        )

    tail = "\n".join(proc.stdout.splitlines()[-15:])
    passed_count = sum(int(n) for n in re.findall(r"(\d+) passed", proc.stdout))
    failed_count = sum(
        int(n) for n in re.findall(r"(\d+) (?:failed|error)", proc.stdout)
    )
    total = passed_count + failed_count
    # exit code 5 = no tests collected; treat as failure when tests were expected
    all_pass = proc.returncode == 0
    score = passed_count / total if total else (1.0 if all_pass else 0.0)
    return EvalResult(
        name="pydev_tests",
        passed=all_pass if spec.require_all_pass else passed_count > 0,
        score=score,
        required=spec.required,
        details={
            "returncode": proc.returncode,
            "passed": passed_count,
            "failed": failed_count,
            "summary": tail,
        },
    )
