"""Persistence layout for run artifacts under ``runs/``.

::

    runs/<task-id>-<UTC-timestamp>/
      meta.json
      control/trial_0/raw.ndjson
      treatment/trial_0/{raw.ndjson, workspace/ (with --keep-workspace)}
      ab_report.json | run_report.json
      report.md
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from skill_eval_kit import __version__
from skill_eval_kit.models.results import ABReport

AB_REPORT_NAME = "ab_report.json"
RUN_REPORT_NAME = "run_report.json"


def new_run_dir(base: Path, task_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"{task_id}-{stamp}"
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{task_id}-{stamp}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_json(model: BaseModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def save_meta(run_dir: Path, extra: dict[str, Any] | None = None) -> Path:
    meta = {
        "ase_version": __version__,
        "python": platform.python_version(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **(extra or {}),
    }
    path = run_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_ab_reports(runs_root: Path) -> list[ABReport]:
    """All A/B reports under a runs directory (used by ``ase rank``)."""
    reports: list[ABReport] = []
    if not runs_root.is_dir():
        return reports
    for path in sorted(runs_root.rglob(AB_REPORT_NAME)):
        try:
            reports.append(ABReport.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return reports
