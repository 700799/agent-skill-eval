"""Declarative task schema (``task.yaml``) and loading helpers.

The schema is strict (``extra="forbid"``) so typos in user YAML fail loudly at
load time, and every field beyond the documented core carries a default so the
minimal spec parses unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_skill_eval import config

#: Default linter set for pydev_static (module-level so the Literal type survives).
_DEFAULT_LINTERS: list[Literal["ruff", "mypy"]] = ["ruff"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceSpec(_StrictModel):
    initial_files: dict[str, str] = Field(default_factory=dict)

    @field_validator("initial_files")
    @classmethod
    def _safe_relative_paths(cls, files: dict[str, str]) -> dict[str, str]:
        for rel in files:
            path = Path(rel)
            if path.is_absolute() or rel.startswith("~"):
                raise ValueError(f"initial_files path must be relative: {rel!r}")
            if ".." in path.parts:
                raise ValueError(f"initial_files path must not contain '..': {rel!r}")
        return files


class RunLimits(_StrictModel):
    max_turns: int = Field(default=10, ge=1)
    max_budget_usd: float = Field(default=0.25, gt=0)
    timeout_seconds: int = Field(default=600, ge=10)
    model: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)


class SkillActivationEval(_StrictModel):
    """Did the agent actually invoke the target skill (vs. generic bypass)?"""

    type: Literal["skill_activation"]
    expected: str
    required: bool = True


class PydevStaticEval(_StrictModel):
    """Run ruff and/or mypy over the final workspace."""

    type: Literal["pydev_static"]
    linters: list[Literal["ruff", "mypy"]] = Field(
        default_factory=lambda: list(_DEFAULT_LINTERS)
    )
    strict: bool = False
    paths: list[str] = Field(default_factory=lambda: ["."])
    required: bool = True


class PydevTestsEval(_StrictModel):
    """Run pytest inside the final workspace."""

    type: Literal["pydev_tests"]
    test_paths: list[str] = Field(default_factory=lambda: ["."])
    require_all_pass: bool = True
    required: bool = True


class AstAssertion(_StrictModel):
    file: str
    kind: Literal[
        "decorator_used",
        "call_used",
        "import_used",
        "class_inherits",
        "function_defined",
        "forbidden_call",
        "forbidden_import",
        "forbidden_decorator",
    ]
    value: str
    count_min: int = Field(default=1, ge=1)


class AstAssertionsEval(_StrictModel):
    """Assert structural patterns in final code (skill-adherence signal)."""

    type: Literal["ast_assertions"]
    assertions: list[AstAssertion] = Field(min_length=1)
    required: bool = True


class TrajectoryEfficiencyEval(_StrictModel):
    """Token-waste / loop anti-pattern thresholds over the trajectory."""

    type: Literal["trajectory_efficiency"]
    max_duplicate_tool_calls: int = Field(default=1, ge=0)
    max_file_re_reads: int = Field(default=2, ge=0)
    max_context_growth_tokens: int = Field(default=15000, ge=1)
    exempt_tools: list[str] = Field(default_factory=lambda: ["TodoWrite"])
    required: bool = True


class LlmJudgeEval(_StrictModel):
    """Free-form rubric judged by a Claude model; returns pass + 0..1 score."""

    type: Literal["llm_judge"]
    rubric: str = Field(min_length=1)
    model: str = config.DEFAULT_JUDGE_MODEL
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    required: bool = True


EvaluationSpec = Annotated[
    SkillActivationEval
    | PydevStaticEval
    | PydevTestsEval
    | AstAssertionsEval
    | TrajectoryEfficiencyEval
    | LlmJudgeEval,
    Field(discriminator="type"),
]


class TaskSpec(_StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    description: str = ""
    target_skill: str | None = None
    workspace: WorkspaceSpec = Field(default_factory=WorkspaceSpec)
    prompt: str = Field(min_length=1)
    run_limits: RunLimits = Field(default_factory=RunLimits)
    evaluations: list[EvaluationSpec] = Field(min_length=1)
    trials: int | None = Field(default=None, ge=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _skill_activation_needs_target(self) -> TaskSpec:
        needs_skill = any(e.type == "skill_activation" for e in self.evaluations)
        if needs_skill and not self.target_skill:
            raise ValueError("a skill_activation evaluation requires target_skill to be set")
        return self


def derive_skill_name(skill_path: Path) -> str:
    """``skills/foo.md`` -> ``foo``; ``skills/foo/SKILL.md`` -> ``foo``."""
    if skill_path.name == "SKILL.md":
        return skill_path.parent.name
    return skill_path.stem


@dataclass(frozen=True)
class LoadedTask:
    """A validated task plus filesystem context resolved from its YAML path."""

    spec: TaskSpec
    path: Path
    skill_path: Path | None
    skill_name: str | None


def load_task(path: str | Path) -> LoadedTask:
    task_path = Path(path).resolve()
    with open(task_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{task_path}: task file must contain a YAML mapping")
    spec = TaskSpec.model_validate(raw)

    skill_path: Path | None = None
    skill_name: str | None = None
    if spec.target_skill:
        skill_path = (task_path.parent / spec.target_skill).resolve()
        if not skill_path.is_file():
            raise FileNotFoundError(f"{task_path}: target_skill not found: {skill_path}")
        skill_name = derive_skill_name(skill_path)
    return LoadedTask(spec=spec, path=task_path, skill_path=skill_path, skill_name=skill_name)
