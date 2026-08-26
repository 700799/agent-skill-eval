"""Replay recorded NDJSON streams — offline tests, CI, and `ase report`."""

from __future__ import annotations

import shutil
from pathlib import Path

from agent_skill_eval.models.task import RunLimits
from agent_skill_eval.models.trajectory import Trajectory
from agent_skill_eval.runner.ndjson import parse_file


class ReplayRunner:
    """Serves fixture files round-robin across successive ``run()`` calls."""

    def __init__(self, fixtures: list[Path]) -> None:
        if not fixtures:
            raise ValueError("ReplayRunner needs at least one fixture file")
        missing = [str(f) for f in fixtures if not Path(f).is_file()]
        if missing:
            raise FileNotFoundError(f"replay fixtures not found: {missing}")
        self.fixtures = [Path(f) for f in fixtures]
        self._calls = 0

    def run(
        self,
        prompt: str,
        workspace: Path,
        limits: RunLimits,
        *,
        bare: bool = False,
        raw_out: Path | None = None,
    ) -> Trajectory:
        del prompt, workspace, limits, bare
        fixture = self.fixtures[self._calls % len(self.fixtures)]
        self._calls += 1
        if raw_out is not None:
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture, raw_out)
        trajectory = parse_file(fixture, source="replay")
        trajectory.outcome.duration_seconds = 0.0
        return trajectory
