"""Detect a usable entry point in an extracted zip bundle that has no method.yaml.

Used as a fallback when a raw ML/KTC repository zip is uploaded without a
manifest — see ``app.py``'s zip upload handler.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Tuple

CONTRACT_CLI_SCRIPT = "cli_script"
CONTRACT_METHOD_PLUGIN = "method_plugin"

_PRIORITY_ENTRY_NAMES = ["main.py", "run.py", "reconstruct.py", "solve.py"]
_SKIP_DIR_NAMES = {"tests", "test", "examples", "docs", "__pycache__"}


def detect_entry_point(extracted_dir: Path) -> Tuple[Path, str]:
    """Find a usable entry point in an extracted bundle that has no method.yaml.

    Checks a priority list of common filenames at the bundle root first
    (``main.py``, ``run.py``, ``reconstruct.py``, ``solve.py``), then falls
    back to every other ``.py`` file directly under the root (subdirectories
    are not scanned). Each candidate is parsed with ``ast`` and classified as:

      - ``CONTRACT_CLI_SCRIPT``: has an ``if __name__ == "__main__":`` guard
        AND uses argparse (``ArgumentParser`` / ``add_argument``) or reads
        ``sys.argv`` directly. Must be run as a subprocess.
      - ``CONTRACT_METHOD_PLUGIN``: defines a class whose base class name
        contains "Plugin" or "Method" and that has a ``reconstruct`` method.
        Can be imported and called in-process.

    The first candidate (in priority order, then alphabetical) that matches
    either contract wins.

    Args:
        extracted_dir: Root of an already-extracted bundle.

    Returns:
        A ``(entry_point_path, contract_type)`` tuple, where contract_type
        is ``CONTRACT_CLI_SCRIPT`` or ``CONTRACT_METHOD_PLUGIN``.

    Raises:
        FileNotFoundError: if no candidate file matches either contract.
    """
    candidates: list[Path] = []
    for name in _PRIORITY_ENTRY_NAMES:
        candidate = extracted_dir / name
        if candidate.is_file():
            candidates.append(candidate)

    root_py_files = sorted(
        p for p in extracted_dir.iterdir()
        if p.is_file() and p.suffix == ".py" and p.parent.name not in _SKIP_DIR_NAMES
    )
    for path in root_py_files:
        if path not in candidates:
            candidates.append(path)

    scanned: list[str] = []
    for path in candidates:
        scanned.append(path.name)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue

        if _is_cli_script(tree):
            return path, CONTRACT_CLI_SCRIPT
        if _is_method_plugin_module(tree):
            return path, CONTRACT_METHOD_PLUGIN

    raise FileNotFoundError(
        f"No usable entry point found in '{extracted_dir}'. "
        f"Scanned: {scanned if scanned else '<no .py files found>'}"
    )


def _is_dunder_main_guard(test: ast.expr) -> bool:
    """True for `__name__ == "__main__"` in either operand order."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False

    left, right = test.left, test.comparators[0]

    def _is_dunder_name(node: ast.expr) -> bool:
        return (isinstance(node, ast.Name) and node.id == "__name__") or (
            isinstance(node, ast.Attribute) and node.attr == "__name__"
        )

    def _is_main_string(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value == "__main__"

    return (_is_dunder_name(left) and _is_main_string(right)) or (
        _is_main_string(left) and _is_dunder_name(right)
    )


def _has_main_guard(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.If) and _is_dunder_main_guard(node.test)
        for node in ast.walk(tree)
    )


def _has_argparse_usage(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr in ("ArgumentParser", "add_argument")
        for node in ast.walk(tree)
    )


def _has_sys_argv_access(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "argv"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        for node in ast.walk(tree)
    )


def _is_cli_script(tree: ast.AST) -> bool:
    return _has_main_guard(tree) and (_has_argparse_usage(tree) or _has_sys_argv_access(tree))


def _base_class_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


def _is_method_plugin_class(node: ast.ClassDef) -> bool:
    has_matching_base = any(
        "Plugin" in _base_class_name(base) or "Method" in _base_class_name(base)
        for base in node.bases
    )
    if not has_matching_base:
        return False
    return any(
        isinstance(item, ast.FunctionDef) and item.name == "reconstruct"
        for item in node.body
    )


def _is_method_plugin_module(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ClassDef) and _is_method_plugin_class(node)
        for node in ast.walk(tree)
    )
