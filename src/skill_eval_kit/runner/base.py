"""Runner protocol: anything that turns (prompt, workspace, limits) into a Trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from skill_eval_kit.models.task import RunLimits
from skill_eval_kit.models.trajectory import Trajectory


class AgentRunner(Protocol):
    def run(
        self,
        prompt: str,
        workspace: Path,
        limits: RunLimits,
        *,
        no_skills: bool = False,
        raw_out: Path | None = None,
    ) -> Trajectory:
        """Execute one agent session in ``workspace`` and return its trajectory.

        ``no_skills=True`` is the Control arm: skill loading is disabled outright,
        so the baseline measures the unaided model. ``raw_out``, when given,
        receives the raw NDJSON event stream for later replay and mining.
        """
        ...
