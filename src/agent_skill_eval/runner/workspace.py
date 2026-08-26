"""Sandbox workspaces: materialize initial files, inject the skill under test."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from agent_skill_eval.models.task import WorkspaceSpec, derive_skill_name


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
