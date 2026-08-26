"""End-to-end A/B over replay fixtures: loopy control vs skill-following treatment."""

from pathlib import Path

from agent_skill_eval.ab import run_ab
from agent_skill_eval.models.results import ABReport
from agent_skill_eval.models.task import LoadedTask, load_task
from agent_skill_eval.report import render_ab_report
from agent_skill_eval.runner.replay import ReplayRunner
from agent_skill_eval.runs import load_ab_reports, new_run_dir, save_json, save_meta

LEGACY_APP = '''\
from fastapi import FastAPI
from pydantic import validator

app = FastAPI()


class Item:
    @validator("name")
    def check(cls, v):
        return v


@app.get("/items")
def read_items():
    return {"id": 1, "name": "Item"}
'''

TASK_YAML = """\
id: pydev-pydantic-refactor-02
description: "Refactor to Pydantic v2 models with field_validator."
target_skill: "skills/fastapi-schema.md"

workspace:
  initial_files:
    app.py: |
{app_indented}

prompt: "Refactor app.py to a Pydantic v2 response model."

run_limits:
  max_turns: 6
  max_budget_usd: 0.15

evaluations:
  - type: skill_activation
    expected: "fastapi-schema"
  - type: pydev_static
    linters: [ruff, mypy]
  - type: ast_assertions
    assertions:
      - file: app.py
        kind: decorator_used
        value: field_validator
      - file: app.py
        kind: forbidden_decorator
        value: validator
  - type: trajectory_efficiency
    max_duplicate_tool_calls: 1
    max_file_re_reads: 2
"""


def write_task(root: Path) -> LoadedTask:
    app_indented = "\n".join(f"      {line}" for line in LEGACY_APP.splitlines())
    task_path = root / "task.yaml"
    task_path.write_text(TASK_YAML.format(app_indented=app_indented), encoding="utf-8")
    skill = root / "skills" / "fastapi-schema.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        "---\nname: fastapi-schema\n---\nAlways use Pydantic v2 @field_validator.\n",
        encoding="utf-8",
    )
    return load_task(task_path)


def run_fixture_ab(
    tmp_path: Path, success_fixture: Path, loopy_fixture: Path, out_dir: Path | None = None
) -> ABReport:
    loaded = write_task(tmp_path)
    runners = {
        "control": ReplayRunner([loopy_fixture]),
        "treatment": ReplayRunner([success_fixture]),
    }
    return run_ab(loaded, runners, trials=2, out_dir=out_dir)  # type: ignore[arg-type]


def test_ab_deltas_favor_treatment(
    tmp_path: Path, success_fixture: Path, loopy_fixture: Path
) -> None:
    report = run_fixture_ab(tmp_path, success_fixture, loopy_fixture)
    assert report.control.n == 2 and report.treatment.n == 2
    assert report.skill_name == "fastapi-schema"

    assert report.control.skill_activation_rate == 0.0
    assert report.treatment.skill_activation_rate == 1.0
    assert report.control.pass_rate == 0.0
    assert report.treatment.pass_rate == 1.0

    deltas = report.deltas
    assert deltas["pass_rate_delta"] == 1.0
    assert deltas["token_delta_pct"] < 0  # treatment saves tokens
    assert deltas["cost_delta_usd"] < 0
    assert deltas["duplicate_calls_delta"] == -3.0
    assert deltas["re_reads_delta"] == -2.0
    assert deltas["skill_activation_delta"] == 1.0


def test_replay_rebuilds_final_workspace(
    tmp_path: Path, success_fixture: Path, loopy_fixture: Path
) -> None:
    """Treatment's recorded Write is applied, so adherence checks see real output."""
    report = run_fixture_ab(tmp_path, success_fixture, loopy_fixture)
    treatment = [t for t in report.trials if t.arm == "treatment"][0]
    control = [t for t in report.trials if t.arm == "control"][0]

    def ast_result(trial: object) -> dict:
        return next(
            r.details for r in trial.eval_results if r.name == "ast_assertions"  # type: ignore[attr-defined]
        )

    assert all(a["ok"] for a in ast_result(treatment)["assertions"])
    # control never wrote the file: legacy @validator still present
    assert not all(a["ok"] for a in ast_result(control)["assertions"])


def test_ab_persists_raw_streams_and_reports(
    tmp_path: Path, success_fixture: Path, loopy_fixture: Path
) -> None:
    out_root = new_run_dir(tmp_path / "runs", "pydev-pydantic-refactor-02")
    report = run_fixture_ab(tmp_path, success_fixture, loopy_fixture, out_dir=out_root)
    save_json(report, out_root / "ab_report.json")
    save_meta(out_root, {"runner": "replay"})
    (out_root / "report.md").write_text(render_ab_report(report), encoding="utf-8")

    assert (out_root / "control" / "trial_0" / "raw.ndjson").exists()
    assert (out_root / "treatment" / "trial_1" / "raw.ndjson").exists()
    assert (out_root / "meta.json").exists()

    loaded_reports = load_ab_reports(tmp_path / "runs")
    assert len(loaded_reports) == 1
    assert loaded_reports[0].task_id == "pydev-pydantic-refactor-02"

    markdown = (out_root / "report.md").read_text()
    assert "| pass rate | 0% | 100% | +100pp |" in markdown
    assert "skill activation | 0% | 100%" in markdown
    assert "duplicate tool calls | 3.0 | 0.0" in markdown


def test_trial_pass_requires_success_subtype(
    tmp_path: Path, error_fixture: Path, success_fixture: Path
) -> None:
    loaded = write_task(tmp_path)
    runners = {
        "control": ReplayRunner([error_fixture]),
        "treatment": ReplayRunner([success_fixture]),
    }
    report = run_ab(loaded, runners, trials=1)  # type: ignore[arg-type]
    control_trial = report.trials[0]
    assert control_trial.outcome_subtype == "error_max_turns"
    assert not control_trial.passed
