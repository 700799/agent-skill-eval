from pathlib import Path

import pytest

from agent_skill_eval.models.task import RunLimits, WorkspaceSpec
from agent_skill_eval.runner.claude_cli import ClaudeCliRunner
from agent_skill_eval.runner.replay import ReplayRunner
from agent_skill_eval.runner.workspace import inject_skill, materialize, sandbox

LIMITS = RunLimits(max_turns=6, max_budget_usd=0.15)


def test_materialize_and_nested_paths(tmp_path: Path) -> None:
    spec = WorkspaceSpec(initial_files={"app.py": "print('hi')\n", "pkg/mod.py": "x = 1\n"})
    root = materialize(spec, tmp_path / "ws")
    assert (root / "app.py").read_text() == "print('hi')\n"
    assert (root / "pkg" / "mod.py").read_text() == "x = 1\n"


def test_inject_flat_skill_file(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "fastapi-schema.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("Use Pydantic v2.\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    name = inject_skill(ws, skill)
    assert name == "fastapi-schema"
    installed = ws / ".claude" / "skills" / "fastapi-schema" / "SKILL.md"
    assert installed.read_text() == "Use Pydantic v2.\n"


def test_inject_skill_bundle_copies_assets(tmp_path: Path) -> None:
    bundle = tmp_path / "skills" / "fastapi-schema"
    bundle.mkdir(parents=True)
    (bundle / "SKILL.md").write_text("body\n")
    (bundle / "reference.md").write_text("extra\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    name = inject_skill(ws, bundle / "SKILL.md")
    assert name == "fastapi-schema"
    dest = ws / ".claude" / "skills" / "fastapi-schema"
    assert (dest / "SKILL.md").exists() and (dest / "reference.md").exists()


def test_sandbox_cleanup(tmp_path: Path) -> None:
    spec = WorkspaceSpec(initial_files={"a.py": "pass\n"})
    with sandbox(spec, base_dir=tmp_path) as root:
        assert (root / "a.py").exists()
        kept = root
    assert not kept.exists()


def test_replay_runner_round_robin(
    tmp_path: Path, success_fixture: Path, loopy_fixture: Path
) -> None:
    runner = ReplayRunner([success_fixture, loopy_fixture])
    first = runner.run("p", tmp_path, LIMITS)
    second = runner.run("p", tmp_path, LIMITS)
    third = runner.run("p", tmp_path, LIMITS, raw_out=tmp_path / "raw.ndjson")
    assert first.skill_activations() == ["fastapi-schema"]
    assert second.skill_activations() == []
    assert third.skill_activations() == ["fastapi-schema"]
    assert (tmp_path / "raw.ndjson").exists()


def test_replay_runner_missing_fixture() -> None:
    with pytest.raises(FileNotFoundError):
        ReplayRunner([Path("/nope/missing.ndjson")])


def test_build_argv_treatment_and_control() -> None:
    runner = ClaudeCliRunner()
    limits = RunLimits(
        max_turns=6,
        max_budget_usd=0.15,
        model="claude-sonnet-5",
        allowed_tools=["Read", "Edit"],
        disallowed_tools=["Bash(rm *)"],
    )
    argv = runner._build_argv("do it", limits, bare=False)
    assert argv[:3] == ["claude", "-p", "do it"]
    assert ["--output-format", "stream-json"] == argv[3:5]
    assert "--verbose" in argv
    assert ["--max-turns", "6"] == argv[argv.index("--max-turns") : argv.index("--max-turns") + 2]
    assert "0.15" in argv
    assert ["--model", "claude-sonnet-5"] == argv[argv.index("--model") : argv.index("--model") + 2]
    assert "--bare" not in argv
    assert ["--allowedTools", "Read,Edit"] == (
        argv[argv.index("--allowedTools") : argv.index("--allowedTools") + 2]
    )
    assert ["--disallowedTools", "Bash(rm *)"] == (
        argv[argv.index("--disallowedTools") : argv.index("--disallowedTools") + 2]
    )
    control = runner._build_argv("do it", LIMITS, bare=True)
    assert "--bare" in control
    assert "--model" not in control


def test_missing_cli_binary_raises(tmp_path: Path) -> None:
    runner = ClaudeCliRunner(claude_bin="claude-definitely-not-installed")
    with pytest.raises(RuntimeError, match="not found"):
        runner.run("p", tmp_path, LIMITS)
