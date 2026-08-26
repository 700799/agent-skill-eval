"""Portfolio models: skill inventory, lint, triggers, clusters, scorecard."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Recommendation = Literal["KEEP", "FIX", "MERGE", "RETIRE"]


class SkillMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    description: str = ""
    body: str = ""
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = 0
    referenced_paths: list[str] = Field(default_factory=list)


class LintFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    severity: Literal["error", "warning", "info"]
    message: str


class LintReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    findings: list[LintFinding] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class OverlapCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[str]
    max_similarity: float
    merge_into: str | None = None
    rationale: str = ""


class TriggerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe: str
    intended_skill: str | None = None
    expected: bool
    activated_skills: list[str] = Field(default_factory=list)
    mode: Literal["classifier", "headless"]


class TriggerReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    results: list[TriggerResult] = Field(default_factory=list)


class CriticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    rewrite_suggestions: list[str] = Field(default_factory=list)
    improved_description: str | None = None


class MinedSkillUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    activations: int = 0
    sessions: int = 0
    est_tokens: int = 0
    est_cost_usd: float = 0.0


class MiningReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessions_scanned: int = 0
    skills: list[MinedSkillUsage] = Field(default_factory=list)
    generated_at: str = ""


class AuditReport(BaseModel):
    """Everything `ase audit` produces in one artifact."""

    model_config = ConfigDict(extra="forbid")

    skills_dir: str
    lint: list[LintReport] = Field(default_factory=list)
    clusters: list[OverlapCluster] = Field(default_factory=list)
    triggers: list[TriggerReport] = Field(default_factory=list)
    confusion: dict[str, dict[str, int]] | None = None
    critics: list[CriticReport] = Field(default_factory=list)
    generated_at: str = ""


class ScorecardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    lint_score: float | None = None
    trigger_f1: float | None = None
    ab_score: float | None = None
    critic_score: float | None = None
    mined_activations: int | None = None
    coverage: dict[str, bool] = Field(default_factory=dict)
    composite: float = 0.0
    recommendation: Recommendation = "KEEP"
    merge_with: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PortfolioReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[ScorecardEntry] = Field(default_factory=list)
    clusters: list[OverlapCluster] = Field(default_factory=list)
    confusion: dict[str, dict[str, int]] | None = None
    generated_at: str = ""
