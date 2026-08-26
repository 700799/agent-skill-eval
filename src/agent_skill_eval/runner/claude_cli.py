"""Live runner: drives `claude -p` as a subprocess and captures the event stream.

The Control arm passes ``--disable-slash-commands``, which is what actually
turns skills off: ``--bare`` only skips hooks, LSP, and plugin credentials —
its own help notes that "Skills still resolve via /skill-name", so it would
leak the very thing the baseline is meant to exclude. Treatment loads the skill
injected into the workspace's ``.claude/skills/`` by
:func:`agent_skill_eval.runner.workspace.inject_skill`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agent_skill_eval.models.task import RunLimits
from agent_skill_eval.models.trajectory import Trajectory
from agent_skill_eval.runner.ndjson import parse_events


class ClaudeCliRunner:
    def __init__(
        self,
        *,
        claude_bin: str = "claude",
        permission_mode: str = "acceptEdits",
        extra_args: list[str] | None = None,
    ) -> None:
        """``permission_mode`` defaults to ``acceptEdits`` so headless runs can
        edit files in their disposable sandbox without prompting.
        ``bypassPermissions`` is deliberately not the default: it maps to
        ``--dangerously-skip-permissions``, which the CLI refuses to run as
        root. Pass ``""`` to omit the flag entirely.
        """
        self.claude_bin = claude_bin
        self.permission_mode = permission_mode
        self.extra_args = list(extra_args or [])

    def _build_argv(self, prompt: str, limits: RunLimits, *, no_skills: bool) -> list[str]:
        argv = [
            self.claude_bin,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(limits.max_turns),
            "--max-budget-usd",
            str(limits.max_budget_usd),
        ]
        if self.permission_mode:
            argv += ["--permission-mode", self.permission_mode]
        if limits.model:
            argv += ["--model", limits.model]
        if no_skills:
            # Control arm: the only flag that genuinely stops skills loading.
            argv += ["--disable-slash-commands"]
        if limits.allowed_tools:
            argv += ["--allowedTools", ",".join(limits.allowed_tools)]
        if limits.disallowed_tools:
            argv += ["--disallowedTools", ",".join(limits.disallowed_tools)]
        argv += self.extra_args
        return argv

    def run(
        self,
        prompt: str,
        workspace: Path,
        limits: RunLimits,
        *,
        no_skills: bool = False,
        raw_out: Path | None = None,
    ) -> Trajectory:
        argv = self._build_argv(prompt, limits, no_skills=no_skills)
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.Popen(
                argv,
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"claude CLI not found ({self.claude_bin!r}); install it or use --runner replay"
            ) from exc
        try:
            stdout, stderr = proc.communicate(timeout=limits.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
        duration = time.monotonic() - started

        if raw_out is not None:
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            raw_out.write_text(stdout or "", encoding="utf-8")

        trajectory = parse_events(
            (stdout or "").splitlines(),
            source="live",
            raw_path=str(raw_out) if raw_out else None,
        )
        trajectory.outcome.duration_seconds = duration
        if timed_out:
            trajectory.outcome.subtype = "error_timeout"
            trajectory.outcome.error = f"killed after {limits.timeout_seconds}s"
        elif trajectory.outcome.subtype == "missing_result" and proc.returncode != 0:
            trajectory.outcome.subtype = "error_cli"
            trajectory.outcome.error = (stderr or "").strip()[-2000:] or (
                f"claude exited {proc.returncode} with no result event"
            )
        return trajectory
