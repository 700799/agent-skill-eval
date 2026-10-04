"""The verbatim spec example must parse unchanged; bad specs must fail loudly."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from skill_eval_kit.models import TaskSpec, derive_skill_name, load_task

SPEC_EXAMPLE = """\
id: pydev-fastapi-pydantic-01
description: "Verify that the fastapi-schema skill refactors raw dicts into idiomatic Pydantic v2 models."
target_skill: "skills/fastapi-schema.md"

workspace:
  initial_files:
    app.py: |
      from fastapi import FastAPI
      app = FastAPI()
      @app.get("/items")
      def read_items(): return {"id": 1, "name": "Item"}

prompt: "Refactor app.py to return a Pydantic model using the loaded fastapi-schema skill."

run_limits:
  max_turns: 6
  max_budget_usd: 0.15

evaluations:
  - type: skill_activation
    expected: "fastapi-schema"
  - type: pydev_static
    linters: [ruff, mypy]
    strict: true
  - type: trajectory_efficiency
    max_duplicate_tool_calls: 1
    max_file_re_reads: 2
  - type: llm_judge
    rubric: "Did the code use Pydantic V2 `@field_validator` syntax as required by the skill instead of legacy V1 syntax?"
"""


def write_spec_example(root: Path) -> Path:
    task_path = root / "task.yaml"
    task_path.write_text(SPEC_EXAMPLE, encoding="utf-8")
    skill = root / "skills" / "fastapi-schema.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text("---\nname: fastapi-schema\n---\nUse Pydantic v2.\n", encoding="utf-8")
    return task_path


def test_spec_example_parses_verbatim(tmp_path: Path) -> None:
    loaded = load_task(write_spec_example(tmp_path))
    spec = loaded.spec
    assert spec.id == "pydev-fastapi-pydantic-01"
    assert spec.run_limits.max_turns == 6
    assert spec.run_limits.max_budget_usd == 0.15
    # defaults for extension fields
    assert spec.run_limits.timeout_seconds == 600
    assert spec.trials is None
    assert [e.type for e in spec.evaluations] == [
        "skill_activation",
        "pydev_static",
        "trajectory_efficiency",
        "llm_judge",
    ]
    assert loaded.skill_name == "fastapi-schema"
    assert loaded.skill_path is not None and loaded.skill_path.is_file()


def test_derive_skill_name() -> None:
    assert derive_skill_name(Path("skills/fastapi-schema.md")) == "fastapi-schema"
    assert derive_skill_name(Path("skills/fastapi-schema/SKILL.md")) == "fastapi-schema"


def test_unknown_evaluation_type_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {
                "id": "t-1",
                "prompt": "do it",
                "evaluations": [{"type": "made_up_eval"}],
            }
        )


def test_unknown_top_level_field_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {
                "id": "t-1",
                "prompt": "do it",
                "evaluations": [{"type": "pydev_static"}],
                "promt": "typo",
            }
        )


@pytest.mark.parametrize("bad_path", ["/etc/passwd", "../escape.py", "~/x.py"])
def test_absolute_or_escaping_initial_files_rejected(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {
                "id": "t-1",
                "prompt": "do it",
                "workspace": {"initial_files": {bad_path: "print()"}},
                "evaluations": [{"type": "pydev_static"}],
            }
        )


def test_skill_activation_requires_target_skill() -> None:
    with pytest.raises(ValidationError, match="target_skill"):
        TaskSpec.model_validate(
            {
                "id": "t-1",
                "prompt": "do it",
                "evaluations": [{"type": "skill_activation", "expected": "foo"}],
            }
        )


def test_missing_skill_file_raises(tmp_path: Path) -> None:
    task_path = write_spec_example(tmp_path)
    (tmp_path / "skills" / "fastapi-schema.md").unlink()
    with pytest.raises(FileNotFoundError):
        load_task(task_path)
