"""Structural adherence checks over final code, via the Python AST.

The primary "skill adherence vs. generic bypass" signal: a task can assert
that the *specific* constructs a skill prescribes (e.g. Pydantic v2
``@field_validator``) actually appear — and that the legacy patterns the
skill forbids do not — independent of whether tests happen to pass.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from skill_eval_kit.evaluators.base import EvalContext, register
from skill_eval_kit.models.results import EvalResult
from skill_eval_kit.models.task import AstAssertion, AstAssertionsEval


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return None


def _matches(candidate: str, value: str) -> bool:
    if candidate == value:
        return True
    if "." not in value:
        return candidate.rsplit(".", 1)[-1] == value
    return candidate.endswith("." + value)


@dataclass
class _Inventory:
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)


def _collect(tree: ast.AST) -> _Inventory:
    inv = _Inventory()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inv.functions.append(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                name = _dotted(decorator)
                if name:
                    inv.decorators.append(name)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = _dotted(base)
                if name:
                    inv.bases.append(name)
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                inv.calls.append(name)
        if isinstance(node, ast.Import):
            inv.imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            inv.imports.append(node.module)
            inv.imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return inv


_KIND_TO_POOL = {
    "decorator_used": "decorators",
    "forbidden_decorator": "decorators",
    "call_used": "calls",
    "forbidden_call": "calls",
    "import_used": "imports",
    "forbidden_import": "imports",
    "class_inherits": "bases",
    "function_defined": "functions",
}


def _check(assertion: AstAssertion, inv: _Inventory) -> tuple[bool, int]:
    pool: list[str] = getattr(inv, _KIND_TO_POOL[assertion.kind])
    count = sum(1 for candidate in pool if _matches(candidate, assertion.value))
    if assertion.kind.startswith("forbidden_"):
        return count == 0, count
    return count >= assertion.count_min, count


@register("ast_assertions")
def evaluate(spec: AstAssertionsEval, ctx: EvalContext) -> EvalResult:
    inventories: dict[str, _Inventory | str] = {}
    outcomes: list[dict[str, object]] = []
    satisfied = 0
    for assertion in spec.assertions:
        if assertion.file not in inventories:
            path = Path(ctx.workspace) / assertion.file
            if not path.is_file():
                inventories[assertion.file] = f"file not found: {assertion.file}"
            else:
                try:
                    inventories[assertion.file] = _collect(
                        ast.parse(path.read_text(encoding="utf-8"))
                    )
                except SyntaxError as exc:
                    inventories[assertion.file] = f"syntax error: {exc}"
        inventory = inventories[assertion.file]
        if isinstance(inventory, str):
            outcomes.append(
                {
                    "kind": assertion.kind,
                    "value": assertion.value,
                    "file": assertion.file,
                    "ok": False,
                    "error": inventory,
                }
            )
            continue
        ok, count = _check(assertion, inventory)
        satisfied += ok
        outcomes.append(
            {
                "kind": assertion.kind,
                "value": assertion.value,
                "file": assertion.file,
                "ok": ok,
                "count": count,
            }
        )
    total = len(spec.assertions)
    return EvalResult(
        name="ast_assertions",
        passed=satisfied == total,
        score=satisfied / total if total else 1.0,
        required=spec.required,
        details={"assertions": outcomes},
    )
