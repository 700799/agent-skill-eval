"""Mine real Claude Code session transcripts for in-the-wild skill usage.

Answers the question no synthetic benchmark can: which skills does anyone
actually trigger, how often, and at what cost? Reuses the same NDJSON parser
as live and replay runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent_skill_eval.metrics.tokens import estimate_cost, trajectory_usage
from agent_skill_eval.models.portfolio import MinedSkillUsage, MiningReport
from agent_skill_eval.runner.ndjson import parse_file

TRANSCRIPT_SUFFIXES = {".jsonl", ".ndjson"}


def iter_transcripts(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix in TRANSCRIPT_SUFFIXES:
            found.append(path)
        elif path.is_dir():
            found.extend(
                p
                for suffix in sorted(TRANSCRIPT_SUFFIXES)
                for p in sorted(path.rglob(f"*{suffix}"))
            )
    return sorted(set(found))


def mine_sessions(
    paths: list[str | Path], *, known_skills: list[str] | None = None
) -> MiningReport:
    known = set(known_skills) if known_skills else None
    activations: dict[str, int] = {}
    sessions: dict[str, set[str]] = {}
    tokens_by_skill: dict[str, int] = {}
    cost_by_skill: dict[str, float] = {}
    scanned = 0

    for transcript in iter_transcripts(paths):
        try:
            trajectory = parse_file(transcript, source="mined")
        except (OSError, ValueError):
            continue
        scanned += 1
        names = [n for n in trajectory.skill_activations() if known is None or n in known]
        if not names:
            continue
        usage, _ = trajectory_usage(trajectory)
        cost = trajectory.outcome.total_cost_usd
        if cost is None:
            cost = estimate_cost(usage, trajectory.model)
        # A session's cost is attributed evenly across the skills it loaded.
        share = len(set(names))
        for name in set(names):
            activations[name] = activations.get(name, 0) + names.count(name)
            sessions.setdefault(name, set()).add(str(transcript))
            tokens_by_skill[name] = tokens_by_skill.get(name, 0) + usage.total_tokens // share
            cost_by_skill[name] = cost_by_skill.get(name, 0.0) + cost / share

    skills = [
        MinedSkillUsage(
            skill=name,
            activations=count,
            sessions=len(sessions.get(name, set())),
            est_tokens=tokens_by_skill.get(name, 0),
            est_cost_usd=round(cost_by_skill.get(name, 0.0), 6),
        )
        for name, count in sorted(activations.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    # Skills that exist but never fired are the point of the exercise: keep them at zero.
    if known:
        seen = {s.skill for s in skills}
        skills.extend(MinedSkillUsage(skill=name) for name in sorted(known - seen))
    return MiningReport(
        sessions_scanned=scanned,
        skills=skills,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
