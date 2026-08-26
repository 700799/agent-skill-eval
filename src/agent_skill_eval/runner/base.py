"""Runner protocol: anything that turns (prompt, workspace, limits) into a Trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agent_skill_eval.models.task import RunLimits
from agent_skill_eval.models.trajectory import Trajectory


class AgentRunner(Protocol):
    def run(
        self,
        prompt: str,
        workspace: Path,
        limits: RunLimits,
        *,
        bare: bool = False,
        raw_out: Path | None = None,
    ) -> Trajectory:
        """Execute one agent session in ``workspace`` and return its trajectory.

        ``bare=True`` is the Control arm: no skills or user settings are loaded.
        ``raw_out``, when given, receives the raw NDJSON event stream for
        later replay and mining.
        """
        ...
