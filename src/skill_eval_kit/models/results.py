"""Result models: per-evaluator, per-trial, per-arm aggregates, A/B reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    required: bool = True


class TrialMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False
    duration_seconds: float = 0.0
    num_turns: int = 0
    duplicate_tool_calls: int = 0
    file_re_reads: int = 0
    max_context_growth: int = 0
    context_growth_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    arm: Literal["control", "treatment"]
    trial_index: int
    outcome_subtype: str
    skill_activated: bool = False
    metrics: TrialMetrics
    eval_results: list[EvalResult] = Field(default_factory=list)
    passed: bool
    ndjson_path: str | None = None


class ArmAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: Literal["control", "treatment"]
    n: int
    pass_rate: float
    mean_total_tokens: float
    median_total_tokens: float
    mean_cost_usd: float
    mean_duration_seconds: float
    mean_duplicate_calls: float
    mean_re_reads: float
    skill_activation_rate: float
    any_cost_estimated: bool = False


class ABReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    skill_name: str | None = None
    control: ArmAggregate
    treatment: ArmAggregate
    #: treatment - control; negative token/cost/duration delta = treatment saves.
    deltas: dict[str, float] = Field(default_factory=dict)
    trials: list[TrialResult] = Field(default_factory=list)
    generated_at: str = ""


class TaskReport(BaseModel):
    """Single-arm report produced by ``ase run``."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    skill_name: str | None = None
    arm: Literal["control", "treatment"]
    trials: list[TrialResult] = Field(default_factory=list)
    generated_at: str = ""
