"""Sandbox workspaces: materialize initial files, inject the skill under test."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from agent_skill_eval.models.task import WorkspaceSpec, derive_skill_name
from agent_skill_eval.models.trajectory import Trajectory


def materialize(spec: WorkspaceSpec, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in spec.initial_files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def inject_skill(root: Path, skill_path: Path) -> str:
    """Install the skill under test into ``<root>/.claude/skills/<name>/SKILL.md``.

    A bare ``foo.md`` becomes ``skills/foo/SKILL.md``; a ``foo/SKILL.md`` bundle
    is copied with its sibling assets. Returns the derived skill name.
    """
    skill_path = skill_path.resolve()
    name = derive_skill_name(skill_path)
    dest = root / ".claude" / "skills" / name
    dest.mkdir(parents=True, exist_ok=True)
    if skill_path.name == "SKILL.md":
        shutil.copytree(skill_path.parent, dest, dirs_exist_ok=True)
    else:
        shutil.copyfile(skill_path, dest / "SKILL.md")
    return name


@contextmanager
def sandbox(
    spec: WorkspaceSpec,
    *,
    skill_path: Path | None = None,
    keep: bool = False,
    base_dir: Path | None = None,
) -> Iterator[Path]:
    """Temp workspace with initial files (and, for the Treatment arm, the skill)."""
    root = Path(tempfile.mkdtemp(prefix="ase-ws-", dir=base_dir))
    try:
        materialize(spec, root)
        if skill_path is not None:
            inject_skill(root, skill_path)
        yield root
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


def _resolve_recorded_path(raw: str, workspace: Path, recorded_cwd: str | None) -> Path | None:
    """Map a path recorded in a trajectory onto the replay workspace.

    Returns None when the path cannot be placed safely inside ``workspace``.
    """
    candidate = PurePosixPath(raw.replace("\\", "/"))
    if candidate.is_absolute():
        base = PurePosixPath((recorded_cwd or "/").replace("\\", "/"))
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            # Recorded outside the session cwd: keep the name only.
            relative = PurePosixPath(candidate.name)
    else:
        relative = candidate
    target = (workspace / relative).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        return None
    return target


def apply_tool_mutations(workspace: Path, trajectory: Trajectory) -> list[str]:
    """Replay recorded Write/Edit/MultiEdit calls so replay rebuilds final state.

    Without this a replayed run leaves the workspace at its initial state and
    the code-quality evaluators would grade the *starting* files rather than
    what the agent produced. Mutations that cannot be applied (missing file,
    unmatched edit, unsafe path) are skipped and reported.
    """
    applied: list[str] = []
    for call in trajectory.iter_tool_calls():
        raw_path = call.input.get("file_path") or call.input.get("notebook_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        target = _resolve_recorded_path(raw_path, workspace, trajectory.cwd)
        if target is None:
            continue

        if call.name == "Write":
            content = call.input.get("content")
            if not isinstance(content, str):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            applied.append(f"Write {raw_path}")
            continue

        if call.name in {"Edit", "MultiEdit"}:
            if not target.is_file():
                continue
            edits = call.input.get("edits")
            if call.name == "Edit":
                edits = [call.input]
            if not isinstance(edits, list):
                continue
            text = target.read_text(encoding="utf-8")
            changed = False
            for edit in edits:
                if not isinstance(edit, dict):
                    continue
                old = edit.get("old_string")
                new = edit.get("new_string")
                if not isinstance(old, str) or not isinstance(new, str) or old not in text:
                    continue
                count = -1 if edit.get("replace_all") else 1
                text = text.replace(old, new, count)
                changed = True
            if changed:
                target.write_text(text, encoding="utf-8")
                applied.append(f"{call.name} {raw_path}")
    return applied
