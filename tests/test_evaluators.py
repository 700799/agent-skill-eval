from pathlib import Path

from tests.conftest import FakeLLM

from agent_skill_eval.evaluators import EvalContext, registered_types, run_evaluations
from agent_skill_eval.evaluators.ast_assertions import evaluate as eval_ast
from agent_skill_eval.evaluators.llm_judge import evaluate as eval_judge
from agent_skill_eval.evaluators.pydev_static import evaluate as eval_static
from agent_skill_eval.evaluators.pydev_tests import evaluate as eval_tests
from agent_skill_eval.evaluators.skill_activation import evaluate as eval_skill
from agent_skill_eval.evaluators.trajectory_efficiency import evaluate as eval_traj
from agent_skill_eval.models.task import (
    AstAssertionsEval,
    LlmJudgeEval,
    PydevStaticEval,
    PydevTestsEval,
    SkillActivationEval,
    TaskSpec,
    TrajectoryEfficiencyEval,
)
from agent_skill_eval.runner.ndjson import parse_file

TASK = TaskSpec(
    id="t-eval",
    prompt="p",
    target_skill="skills/fastapi-schema.md",
    evaluations=[{"type": "skill_activation", "expected": "fastapi-schema"}],  # type: ignore[list-item]
)

PYDANTIC_V2_CODE = '''\
from pydantic import BaseModel, field_validator


class Item(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def check(cls, v: str) -> str:
        return v
'''


def ctx_for(fixture: Path, workspace: Path, llm: FakeLLM | None = None) -> EvalContext:
    return EvalContext(
        task=TASK,
        trajectory=parse_file(fixture, source="replay"),
        workspace=workspace,
        skill_name="fastapi-schema",
        llm=llm,
    )


def test_all_six_registered() -> None:
    assert registered_types() == [
        "ast_assertions",
        "llm_judge",
        "pydev_static",
        "pydev_tests",
        "skill_activation",
        "trajectory_efficiency",
    ]


class TestSkillActivation:
    def test_activated(self, success_fixture: Path, tmp_path: Path) -> None:
        spec = SkillActivationEval(type="skill_activation", expected="fastapi-schema")
        result = eval_skill(spec, ctx_for(success_fixture, tmp_path))
        assert result.passed and result.score == 1.0

    def test_bypassed(self, loopy_fixture: Path, tmp_path: Path) -> None:
        spec = SkillActivationEval(type="skill_activation", expected="fastapi-schema")
        result = eval_skill(spec, ctx_for(loopy_fixture, tmp_path))
        assert not result.passed
        assert result.details["activations"] == []


class TestPydevStatic:
    def test_clean_file_passes(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text('X = 1\nprint(X)\n')
        spec = PydevStaticEval(type="pydev_static", linters=["ruff", "mypy"])
        result = eval_static(spec, ctx_for(success_fixture, tmp_path))
        assert result.passed, result.details
        assert result.score == 1.0

    def test_dirty_file_fails_both_linters(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("import os\nx: int = 'not an int'\n")
        spec = PydevStaticEval(type="pydev_static", linters=["ruff", "mypy"])
        result = eval_static(spec, ctx_for(success_fixture, tmp_path))
        assert not result.passed
        details = result.details
        assert details["ruff"]["issues"] >= 1  # F401 unused import
        assert details["mypy"]["issues"] >= 1  # assignment type error


class TestPydevTests:
    def test_passing_suite(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
        spec = PydevTestsEval(type="pydev_tests")
        result = eval_tests(spec, ctx_for(success_fixture, tmp_path))
        assert result.passed and result.details["passed"] == 1

    def test_failing_suite(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "test_bad.py").write_text(
            "def test_ok():\n    assert True\n\ndef test_bad():\n    assert False\n"
        )
        spec = PydevTestsEval(type="pydev_tests")
        result = eval_tests(spec, ctx_for(success_fixture, tmp_path))
        assert not result.passed
        assert result.details == result.details | {"passed": 1, "failed": 1}
        assert result.score == 0.5


class TestAstAssertions:
    def test_adherence_patterns(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(PYDANTIC_V2_CODE)
        spec = AstAssertionsEval(
            type="ast_assertions",
            assertions=[
                {"file": "app.py", "kind": "decorator_used", "value": "field_validator"},  # type: ignore[list-item]
                {"file": "app.py", "kind": "import_used", "value": "pydantic.field_validator"},  # type: ignore[list-item]
                {"file": "app.py", "kind": "class_inherits", "value": "BaseModel"},  # type: ignore[list-item]
                {"file": "app.py", "kind": "forbidden_decorator", "value": "validator"},  # type: ignore[list-item]
                {"file": "app.py", "kind": "forbidden_import", "value": "pydantic.validator"},  # type: ignore[list-item]
            ],
        )
        result = eval_ast(spec, ctx_for(success_fixture, tmp_path))
        assert result.passed, result.details

    def test_legacy_pattern_caught(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            "from pydantic import validator\n\n"
            "class M:\n"
            "    @validator('x')\n"
            "    def check(cls, v): return v\n"
        )
        spec = AstAssertionsEval(
            type="ast_assertions",
            assertions=[
                {"file": "app.py", "kind": "decorator_used", "value": "field_validator"},  # type: ignore[list-item]
                {"file": "app.py", "kind": "forbidden_decorator", "value": "validator"},  # type: ignore[list-item]
            ],
        )
        result = eval_ast(spec, ctx_for(success_fixture, tmp_path))
        assert not result.passed
        assert result.score == 0.0

    def test_missing_file_and_syntax_error(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def (:\n")
        spec = AstAssertionsEval(
            type="ast_assertions",
            assertions=[
                {"file": "nope.py", "kind": "call_used", "value": "f"},  # type: ignore[list-item]
                {"file": "broken.py", "kind": "call_used", "value": "f"},  # type: ignore[list-item]
            ],
        )
        result = eval_ast(spec, ctx_for(success_fixture, tmp_path))
        assert not result.passed
        errors = [a["error"] for a in result.details["assertions"]]
        assert "file not found" in errors[0] and "syntax error" in errors[1]


class TestTrajectoryEfficiency:
    def test_clean_trajectory_passes(self, success_fixture: Path, tmp_path: Path) -> None:
        spec = TrajectoryEfficiencyEval(type="trajectory_efficiency")
        result = eval_traj(spec, ctx_for(success_fixture, tmp_path))
        assert result.passed and result.score == 1.0
        assert not result.details["context_growth_estimated"]

    def test_loopy_trajectory_fails(self, loopy_fixture: Path, tmp_path: Path) -> None:
        spec = TrajectoryEfficiencyEval(
            type="trajectory_efficiency", max_duplicate_tool_calls=1, max_file_re_reads=2
        )
        result = eval_traj(spec, ctx_for(loopy_fixture, tmp_path))
        assert not result.passed
        assert result.details["duplicate_tool_calls"] == 3
        assert result.details["file_re_reads"] == 2
        assert result.details["context_growth_violations"]
        assert result.details["context_growth_estimated"]


class TestLlmJudge:
    SPEC = LlmJudgeEval(type="llm_judge", rubric="Used Pydantic v2 @field_validator?")

    def test_pass_verdict(self, success_fixture: Path, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(PYDANTIC_V2_CODE)
        llm = FakeLLM([{"passed": True, "score": 0.95, "reasoning": "v2 syntax used"}])
        result = eval_judge(self.SPEC, ctx_for(success_fixture, tmp_path, llm))
        assert result.passed and result.score == 0.95
        assert "field_validator" in llm.calls[0]["user"]
        assert llm.calls[0]["model"] == "claude-opus-5"

    def test_below_threshold_fails(self, success_fixture: Path, tmp_path: Path) -> None:
        llm = FakeLLM([{"passed": True, "score": 0.5, "reasoning": "partial"}])
        result = eval_judge(self.SPEC, ctx_for(success_fixture, tmp_path, llm))
        assert not result.passed

    def test_offline_skips_without_gating(self, success_fixture: Path, tmp_path: Path) -> None:
        result = eval_judge(self.SPEC, ctx_for(success_fixture, tmp_path, None))
        assert not result.passed
        assert not result.required  # skipped judge never gates the trial
        assert "skipped" in result.details


def test_run_evaluations_dispatch(success_fixture: Path, tmp_path: Path) -> None:
    results = run_evaluations(TASK, ctx_for(success_fixture, tmp_path))
    assert [r.name for r in results] == ["skill_activation"]
    assert results[0].passed
