"""Live tests against the real claude CLI. Deselected by default (`-m live`).

These exist because fixtures cannot verify the one thing that actually drifts:
the CLI's real flag surface and stream shape.
"""

import shutil
from pathlib import Path

import pytest

from agent_skill_eval.cli import EXIT_OK, main
from agent_skill_eval.models.task import RunLimits
from agent_skill_eval.runner.claude_cli import ClaudeCliRunner

pytestmark = pytest.mark.live

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
HELLO = EXAMPLES / "tasks" / "pydev-hello-01.yaml"


@pytest.fixture(autouse=True)
def _require_cli() -> None:
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")


def test_stream_shape_still_parses(tmp_path: Path) -> None:
    """The parser must survive the real event stream, extras and all."""
    runner = ClaudeCliRunner()
    limits = RunLimits(max_turns=1, max_budget_usd=0.5, timeout_seconds=180)
    trajectory = runner.run(
        "Say OK and stop.", tmp_path, limits, raw_out=tmp_path / "raw.ndjson"
    )
    assert trajectory.outcome.subtype == "success", trajectory.outcome.error
    assert trajectory.model  # from the system:init event, not clobbered by other subtypes
    assert trajectory.cwd == str(tmp_path)
    assert trajectory.turns
    # live runs carry per-message usage, so accounting is exact rather than estimated
    assert all(turn.usage is not None for turn in trajectory.turns)
    assert trajectory.outcome.total_cost_usd is not None
    assert (tmp_path / "raw.ndjson").is_file()


def test_control_arm_flag_is_accepted() -> None:
    """--disable-slash-commands (not --bare) is what turns skills off."""
    argv = ClaudeCliRunner()._build_argv("hi", RunLimits(), no_skills=True)
    assert "--disable-slash-commands" in argv
    assert "--bare" not in argv


def test_live_treatment_run_edits_workspace(tmp_path: Path) -> None:
    code = main(
        [
            "--out-dir", str(tmp_path / "runs"),
            "--no-llm",
            "run", str(HELLO),
            "--arm", "treatment",
            "--max-budget-usd", "0.6",
            "--keep-workspace",
        ]
    )
    assert code == EXIT_OK
    run_dir = next((tmp_path / "runs").iterdir())
    workspace = run_dir / "treatment" / "trial_0" / "workspace"
    assert (workspace / ".claude" / "skills" / "fastapi-schema" / "SKILL.md").is_file()
    edited = (workspace / "hello.py").read_text()
    # The agent annotated greet(); exact phrasing varies run to run.
    assert "->" in edited or "name: str" in edited
    assert not (workspace / ".ruff_cache").exists()  # evaluators leave no litter
