"""Discover and parse SKILL.md files across a skill library."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from skill_eval_kit.metrics.tokens import estimate_tokens
from skill_eval_kit.models.portfolio import SkillMeta

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
#: Relative path-like tokens in the body (references/foo.md, scripts/run.py, ...).
_PATH_TOKEN = re.compile(r"(?<![\w./-])([\w-]+(?:/[\w.-]+)+\.[A-Za-z0-9]{1,5})")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}


def parse_skill(path: Path, *, name: str | None = None) -> SkillMeta:
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter: dict[str, Any] = {}
    body = raw
    match = _FRONTMATTER.match(raw)
    if match:
        try:
            loaded = yaml.safe_load(match.group(1))
            if isinstance(loaded, dict):
                frontmatter = loaded
        except yaml.YAMLError:
            frontmatter = {}
        body = raw[match.end() :]

    if name is None:
        name = path.parent.name if path.name == "SKILL.md" else path.stem
    declared = frontmatter.get("name")
    description = frontmatter.get("description")
    return SkillMeta(
        name=str(declared) if isinstance(declared, str) and declared else name,
        path=str(path),
        description=str(description) if isinstance(description, str) else "",
        body=body,
        frontmatter=frontmatter,
        token_estimate=estimate_tokens(raw),
        referenced_paths=sorted(set(_PATH_TOKEN.findall(body))),
    )


def discover_skills(root: str | Path) -> list[SkillMeta]:
    """Find every skill under ``root``.

    Supports both layouts: bundle directories (``<name>/SKILL.md``) and flat
    files (``<name>.md``). Plugin trees nest skills arbitrarily deep, so the
    search is recursive.
    """
    root_path = Path(root)
    if root_path.is_file():
        return [parse_skill(root_path)]
    if not root_path.is_dir():
        raise FileNotFoundError(f"skills directory not found: {root_path}")

    found: dict[str, SkillMeta] = {}
    for path in sorted(root_path.rglob("SKILL.md")):
        if _SKIP_DIRS & set(path.parts):
            continue
        meta = parse_skill(path)
        found[meta.name] = meta
    for path in sorted(root_path.rglob("*.md")):
        if path.name == "SKILL.md" or _SKIP_DIRS & set(path.parts):
            continue
        if path.name.upper() in {"README.MD", "CHANGELOG.MD", "LICENSE.MD"}:
            continue
        meta = parse_skill(path)
        found.setdefault(meta.name, meta)
    return sorted(found.values(), key=lambda s: s.name)


def skill_dir(meta: SkillMeta) -> Path:
    path = Path(meta.path)
    return path.parent if path.name == "SKILL.md" else path.parent
