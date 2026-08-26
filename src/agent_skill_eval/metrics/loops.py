"""Token-burning anti-pattern detectors, as pure functions over a Trajectory.

Three detectors:

* duplicate tool calls — the same tool invoked with byte-identical input;
* large-file re-reads — a >100-line file Read again with no intervening
  edit to that file (an edit legitimately resets the window);
* context growth — tokens appended to the context per assistant turn,
  exact when per-message usage is present, chars/4 estimated otherwise.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass, field
from typing import Any

from agent_skill_eval import config
from agent_skill_eval.metrics.tokens import estimate_tokens
from agent_skill_eval.models.trajectory import ToolCall, Trajectory

MUTATING_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _canonical_input_key(call: ToolCall) -> tuple[str, str]:
    canonical = json.dumps(
        call.input, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return call.name, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return posixpath.normpath(value.replace("\\", "/"))


def _call_path(call: ToolCall) -> str | None:
    return _normalize_path(call.input.get("file_path") or call.input.get("notebook_path"))


@dataclass
class DuplicateReport:
    duplicate_tool_calls: int = 0
    worst_offenders: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReReadReport:
    file_re_reads: int = 0
    per_path: dict[str, int] = field(default_factory=dict)


@dataclass
class ContextGrowthReport:
    max_growth: int = 0
    estimated: bool = False
    violations: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class LoopMetrics:
    duplicates: DuplicateReport
    re_reads: ReReadReport
    context_growth: ContextGrowthReport


def detect_duplicates(
    trajectory: Trajectory, exempt_tools: list[str] | None = None
) -> DuplicateReport:
    exempt = set(exempt_tools if exempt_tools is not None else ["TodoWrite"])
    counts: dict[tuple[str, str], int] = {}
    previews: dict[tuple[str, str], str] = {}
    for call in trajectory.iter_tool_calls():
        if call.name in exempt:
            continue
        key = _canonical_input_key(call)
        counts[key] = counts.get(key, 0) + 1
        if key not in previews:
            preview = json.dumps(call.input, ensure_ascii=False, default=str)
            previews[key] = preview[:120]
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    offenders = [
        {"tool": key[0], "count": count, "input_preview": previews[key]}
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1])
        if count >= 3
    ]
    return DuplicateReport(duplicate_tool_calls=duplicates, worst_offenders=offenders)


def detect_re_reads(
    trajectory: Trajectory, large_file_lines: int = config.LARGE_FILE_LINES
) -> ReReadReport:
    window_reads: dict[str, int] = {}
    re_reads: dict[str, int] = {}
    for call in trajectory.iter_tool_calls():
        path = _call_path(call)
        if path is None:
            continue
        if call.name in MUTATING_TOOLS:
            window_reads[path] = 0
            continue
        if call.name != "Read":
            continue
        result = trajectory.result_for(call)
        if result is not None and result.content_text:
            line_count = result.line_count
        else:
            limit = call.input.get("limit")
            # No result and no explicit limit: assume large (whole-file read).
            line_count = int(limit) if isinstance(limit, int) else large_file_lines + 1
        if line_count <= large_file_lines:
            continue
        window_reads[path] = window_reads.get(path, 0) + 1
        if window_reads[path] > 1:
            re_reads[path] = re_reads.get(path, 0) + 1
    return ReReadReport(file_re_reads=sum(re_reads.values()), per_path=re_reads)


def detect_context_growth(
    trajectory: Trajectory, threshold_tokens: int = 15000
) -> ContextGrowthReport:
    report = ContextGrowthReport()
    turns = trajectory.turns
    inter = trajectory.inter_turn_char_lens
    for i in range(1, len(turns)):
        current, previous = turns[i], turns[i - 1]
        if current.usage is not None and previous.usage is not None:
            growth = current.usage.context_tokens - previous.usage.context_tokens
        else:
            inter_chars = inter[i] if i < len(inter) else 0
            growth = estimate_tokens(previous.char_len + inter_chars)
            report.estimated = True
        report.max_growth = max(report.max_growth, growth)
        if growth > threshold_tokens:
            report.violations.append((i, growth))
    return report


def analyze(
    trajectory: Trajectory,
    *,
    exempt_tools: list[str] | None = None,
    large_file_lines: int = config.LARGE_FILE_LINES,
    context_threshold_tokens: int = 15000,
) -> LoopMetrics:
    return LoopMetrics(
        duplicates=detect_duplicates(trajectory, exempt_tools),
        re_reads=detect_re_reads(trajectory, large_file_lines),
        context_growth=detect_context_growth(trajectory, context_threshold_tokens),
    )
