"""CLI surface: exit codes, artifacts, and the full offline audit -> rank flow."""

import json
from pathlib import Path

import pytest

from skill_eval_kit.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TASK = EXAMPLES / "tasks" / "pydev-fastapi-pydantic-01.yaml"


class TestValidate:
    def test_examples_all_valid(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "validate",
                str(TASK),
                str(EXAMPLES / "tasks" / "pydev-hello-01.yaml"),
                str(EXAMPLES / "probes.yaml"),
                str(EXAMPLES / "skills" / "fastapi-schema" / "SKILL.md"),
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert out.count("OK ") == 4

    def test_broken_task_reports_and_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("id: nope\nprompt: p\nevaluations: []\n", encoding="utf-8")
        code = main(["validate", str(bad)])
        assert code == EXIT_FAILED
        assert "INVALID" in capsys.readouterr().out

    def test_missing_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["validate", "/nope/absent.yaml"]) == EXIT_FAILED
        assert "MISSING" in capsys.readouterr().out


class TestRunAndAb:
    def test_run_treatment_replay(
        self, tmp_path: Path, success_fixture: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "--out-dir", str(tmp_path / "runs"),
                "--no-llm",
                "run", str(TASK),
                "--runner", "replay",
                "--fixture", str(success_fixture),
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK, out
        assert "Run report" in out and "skill_activation" in out
        run_dirs = list((tmp_path / "runs").iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "run_report.json").is_file()
        assert (run_dirs[0] / "treatment" / "trial_0" / "raw.ndjson").is_file()

    def test_run_control_arm_fails_on_loopy_fixture(
        self, tmp_path: Path, loopy_fixture: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "--out-dir", str(tmp_path / "runs"),
                "--no-llm",
                "run", str(TASK),
                "--arm", "control",
                "--runner", "replay",
                "--fixture", str(loopy_fixture),
            ]
        )
        assert code == EXIT_FAILED
        assert "FAIL" in capsys.readouterr().out

    def test_ab_writes_reports_and_json(
        self,
        tmp_path: Path,
        success_fixture: Path,
        loopy_fixture: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        json_out = tmp_path / "ab.json"
        code = main(
            [
                "--out-dir", str(tmp_path / "runs"),
                "--no-llm",
                "ab", str(TASK),
                "--runner", "replay",
                "--trials", "2",
                "--control-fixture", str(loopy_fixture),
                "--treatment-fixture", str(success_fixture),
                "--json", str(json_out),
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "A/B report" in out and "pass rate" in out
        payload = json.loads(json_out.read_text())
        assert payload["skill_name"] == "fastapi-schema"
        assert payload["deltas"]["pass_rate_delta"] == 1.0
        assert payload["deltas"]["token_delta_pct"] < 0

    def test_replay_without_fixture_is_a_usage_error(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "--out-dir", str(tmp_path),
                    "run", str(TASK), "--runner", "replay",
                ]
            )

    def test_report_rerenders_stored_run(
        self,
        tmp_path: Path,
        success_fixture: Path,
        loopy_fixture: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(
            [
                "--out-dir", str(tmp_path / "runs"),
                "--no-llm",
                "ab", str(TASK),
                "--runner", "replay",
                "--trials", "1",
                "--control-fixture", str(loopy_fixture),
                "--treatment-fixture", str(success_fixture),
            ]
        )
        capsys.readouterr()
        run_dir = next((tmp_path / "runs").iterdir())
        assert main(["report", str(run_dir)]) == EXIT_OK
        assert "A/B report" in capsys.readouterr().out
        assert main(["report", str(run_dir), "--format", "json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["task_id"] == "pydev-fastapi-pydantic-01"

    def test_report_on_empty_dir(self, tmp_path: Path) -> None:
        assert main(["report", str(tmp_path)]) == EXIT_USAGE


class TestPortfolioCommands:
    def test_audit_json_and_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        audit_json = tmp_path / "audit.json"
        code = main(
            ["--no-llm", "audit", str(EXAMPLES / "skills"), "--json", str(audit_json)]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "dead-reference" in out and "Overlap clusters" in out
        payload = json.loads(audit_json.read_text())
        assert {r["skill"] for r in payload["lint"]} == {
            "fastapi-schema",
            "helpful-helper",
            "py-validation",
        }
        assert payload["clusters"][0]["skills"] == ["fastapi-schema", "py-validation"]

    def test_mine_reports_zero_usage_skills(
        self, tmp_path: Path, mined_fixture: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mine_json = tmp_path / "mine.json"
        code = main(
            [
                "mine", str(mined_fixture.parent),
                "--skills-dir", str(EXAMPLES / "skills"),
                "--json", str(mine_json),
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "fastapi-schema" in out
        payload = json.loads(mine_json.read_text())
        by_skill = {s["skill"]: s for s in payload["skills"]}
        assert by_skill["fastapi-schema"]["activations"] == 1
        assert by_skill["helpful-helper"]["activations"] == 0

    def test_rank_composes_audit_runs_and_mining(
        self,
        tmp_path: Path,
        success_fixture: Path,
        loopy_fixture: Path,
        mined_fixture: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runs = tmp_path / "runs"
        audit_json, mine_json = tmp_path / "audit.json", tmp_path / "mine.json"
        leaderboard = tmp_path / "leaderboard.md"

        main(
            [
                "--out-dir", str(runs), "--no-llm",
                "ab", str(TASK),
                "--runner", "replay", "--trials", "2",
                "--control-fixture", str(loopy_fixture),
                "--treatment-fixture", str(success_fixture),
            ]
        )
        main(["--no-llm", "audit", str(EXAMPLES / "skills"), "--json", str(audit_json)])
        main(
            [
                "mine", str(mined_fixture.parent),
                "--skills-dir", str(EXAMPLES / "skills"),
                "--json", str(mine_json),
            ]
        )
        capsys.readouterr()

        code = main(
            [
                "rank", str(EXAMPLES / "skills"),
                "--audit", str(audit_json),
                "--runs", str(runs),
                "--mine", str(mine_json),
                "--md", str(leaderboard),
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "Skill portfolio leaderboard" in out
        markdown = leaderboard.read_text()

        # fastapi-schema has A/B evidence and clean lint: it tops the board.
        assert markdown.index("fastapi-schema") < markdown.index("helpful-helper")
        # three skills, three different calls
        assert "**KEEP**" in markdown
        assert "MERGE" in markdown
        assert "FIX" in markdown or "RETIRE" in markdown
        # coverage gaps are visible rather than silently scored as zero
        assert "—" in markdown

    def test_rank_weight_override(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["rank", str(EXAMPLES / "skills"), "--weights", "lint=0.5,trigger=0.5"]
        )
        assert code == EXIT_OK
        assert "leaderboard" in capsys.readouterr().out

    def test_rank_rejects_unknown_weight(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["rank", str(EXAMPLES / "skills"), "--weights", "bogus=1.0"])

    def test_audit_missing_dir(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["audit", "/nope/skills"]) == EXIT_USAGE
        assert "error:" in capsys.readouterr().err
