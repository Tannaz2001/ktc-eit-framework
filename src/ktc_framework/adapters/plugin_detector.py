"""
src/ktc_framework/adapters/plugin_detector.py
-----------------------------------------------
Classify an uploaded .py file into one of two execution contracts, using
static analysis only (the ``ast`` module) — the file's top-level code is
never executed or imported.

WHY AST-ONLY MATTERS
---------------------
Uploaded files come from unknown third parties. Importing a .py file runs
its top-level code immediately (module-level statements execute at import
time in Python). Parsing it with ``ast.parse`` only builds a syntax tree —
nothing in the file ever runs. This lets us classify a file safely before
deciding whether (and how) to execute it at all.

THE TWO CONTRACTS
-----------------
CONTRACT_INPROCESS
    A class with a ``reconstruct(self, batch)`` method, or a class
    decorated with ``@register_method`` (or the legacy ``@register``
    alias — see ``ensure_method_plugin_registered`` in app.py for why both
    are recognised). The framework can import this file directly and call
    the method in-process; ``batch`` is a live ``DataBatch`` object.

CONTRACT_CLI
    The raw KTC competition submission shape: a top-level ``main()``,
    an ``argparse`` import, and a module-level
    ``if __name__ == "__main__":`` guard. Invoked as
    ``main.py <inputFolder> <outputFolder> <categoryNbr>``, reading
    ``ref.mat`` / ``data*.mat`` from ``inputFolder`` and writing a
    ``reconstruction`` field (256, 256) with labels {0, 1, 2} into a
    ``.mat`` file in ``outputFolder``. This is the contract every KTC2023
    entry (ABC1, CUQI, Denker, ...) ships in. It cannot be imported
    in-process — it must run as a subprocess (see
    ``cli_plugin_wrapper.CLIScriptPlugin``).

Everything else — a script with neither shape — is CONTRACT_UNKNOWN.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Public constants — the three possible return values of detect_contract()
# ---------------------------------------------------------------------------
CONTRACT_INPROCESS = "inprocess"
CONTRACT_CLI = "cli"
CONTRACT_UNKNOWN = "unknown"

# Both decorator spellings the framework treats as "this class is a
# registered method". Kept as a set so both call-forms (@register_method
# and @register_method(...)) and both import styles resolve the same way.
_REGISTER_DECORATOR_NAMES = {"register_method", "register"}

# Best-effort keyword sets used to fuzzy-match the three positional CLI
# arguments (inputFolder, outputFolder, categoryNbr) by name. Order matters:
# position 0 is checked against the first tuple, position 1 against the
# second, position 2 against the third.
_CLI_ARG_KEYWORDS = (
    ("input", "infolder", "indir", "in_dir"),
    ("output", "outfolder", "outdir", "out_dir"),
    ("categ", "level", "class", "difficulty"),
)


class PluginDetectionError(Exception):
    """Raised when an uploaded .py file cannot be read or parsed.

    Deliberately narrow: this is a file-integrity problem (bad encoding,
    invalid syntax), not a classification result. Callers should treat it
    as "reject the upload", distinct from a clean CONTRACT_UNKNOWN verdict.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_contract(file_path) -> str:
    """Classify *file_path* as 'inprocess', 'cli', or 'unknown'.

    In-process detection runs first and wins on ambiguity: a file that
    happens to define both a registered class AND a CLI-shaped main() is
    treated as in-process, since that is the more directly usable contract.

    Parameters
    ----------
    file_path:
        Path to the uploaded .py file (str or Path).

    Returns
    -------
    str
        One of ``CONTRACT_INPROCESS``, ``CONTRACT_CLI``, ``CONTRACT_UNKNOWN``.

    Raises
    ------
    PluginDetectionError
        If the file cannot be read or does not parse as valid Python.
    """
    tree = _parse(file_path)

    if _has_inprocess_contract(tree):
        return CONTRACT_INPROCESS
    if _has_cli_contract(tree):
        return CONTRACT_CLI
    return CONTRACT_UNKNOWN


def has_argparse_signature(file_path) -> bool:
    """Best-effort check that the file's argparse calls define exactly three
    positional arguments, in an order resembling
    ``(inputFolder, outputFolder, categoryNbr)``.

    This is a secondary confidence check on top of ``detect_contract`` —
    a file can be CONTRACT_CLI (has main/argparse/`__main__` guard) while
    still defining an argparse interface that doesn't match the KTC shape
    (e.g. extra flags, wrong argument count). Use this when you need to
    warn the user their CLI script's signature looks non-standard.

    Parameters
    ----------
    file_path:
        Path to the uploaded .py file (str or Path).

    Returns
    -------
    bool
        True if exactly three positional (non-flag) arguments were passed
        to ``add_argument()`` calls, and each one's name loosely matches the
        expected input/output/category role in order.

    Raises
    ------
    PluginDetectionError
        If the file cannot be read or does not parse as valid Python.
    """
    tree = _parse(file_path)
    positional_names = _positional_add_argument_names(tree)

    if len(positional_names) != 3:
        return False

    return all(
        any(keyword in name.lower() for keyword in keywords)
        for name, keywords in zip(positional_names, _CLI_ARG_KEYWORDS)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse(file_path) -> ast.Module:
    """Read and parse *file_path* into an AST, wrapping all failure modes
    (missing file, bad encoding, invalid syntax) into PluginDetectionError
    so callers only need to catch one exception type."""
    path = Path(file_path)

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PluginDetectionError(f"Cannot read '{path}': {exc}") from exc

    try:
        # filename=str(path) makes any downstream SyntaxError message
        # reference the real file, which is friendlier for debugging.
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise PluginDetectionError(f"Cannot parse '{path}': {exc}") from exc


def _walk_in_order(node: ast.AST):
    """Depth-first, left-to-right traversal that preserves source order.

    ``ast.walk`` (stdlib) is a breadth-first traversal via an internal
    deque — fine for existence checks, but it does not guarantee that
    sibling statements come out in the order they appear in the file.
    ``has_argparse_signature`` depends on argument order (input, output,
    category), so it uses this generator instead of ``ast.walk``.
    """
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_in_order(child)


def _is_register_decorator(dec: ast.expr) -> bool:
    """True if *dec* is ``@register_method`` / ``@register`` in either the
    bare-name form (``@register_method``) or the call form
    (``@register_method(...)``), and either as a plain name or an attribute
    access (``@registry.register_method``)."""
    if isinstance(dec, ast.Call):
        dec = dec.func  # unwrap @decorator(...) down to the decorator name
    if isinstance(dec, ast.Name):
        return dec.id in _REGISTER_DECORATOR_NAMES
    if isinstance(dec, ast.Attribute):
        return dec.attr in _REGISTER_DECORATOR_NAMES
    return False


def _has_inprocess_contract(tree: ast.Module) -> bool:
    """True if any class in the file either:
      (a) is decorated with @register_method / @register, or
      (b) defines a `reconstruct` method whose parameters include `batch`
          right after `self` (matching MethodPlugin.reconstruct(self, batch)).
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        if any(_is_register_decorator(dec) for dec in node.decorator_list):
            return True

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "reconstruct":
                arg_names = [a.arg for a in item.args.args]
                # Require "self" first, then "batch" appearing somewhere in
                # the remaining positional args — lenient enough to allow
                # extra trailing args (e.g. reconstruct(self, batch, debug=False)).
                if len(arg_names) >= 2 and arg_names[0] == "self" and "batch" in arg_names[1:]:
                    return True

    return False


def _is_dunder_main_check(test: ast.expr) -> bool:
    """True if *test* is ``__name__ == "__main__"`` (or the reversed
    ``"__main__" == __name__`` — both are valid Python and both appear in
    the wild)."""
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False

    left, right = test.left, test.comparators[0]

    def _is_dunder_name(n: ast.expr) -> bool:
        return isinstance(n, ast.Name) and n.id == "__name__"

    def _is_main_string(n: ast.expr) -> bool:
        return isinstance(n, ast.Constant) and n.value == "__main__"

    return (_is_dunder_name(left) and _is_main_string(right)) or (
        _is_main_string(left) and _is_dunder_name(right)
    )


def _has_cli_contract(tree: ast.Module) -> bool:
    """True if the module has all three markers of the KTC CLI contract:
    a top-level `main()`, an `argparse` import anywhere in the file, and a
    module-level `if __name__ == "__main__":` guard.

    `main` and the `__main__` guard are required at module level (directly
    in `tree.body`) — a `main()` nested inside a class or another function
    doesn't count as the script's CLI entry point.
    """
    has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in tree.body
    )
    if not has_main:
        return False

    # argparse could technically be imported inside a function rather than
    # at module level, so this check walks the whole tree rather than just
    # tree.body.
    has_argparse_import = any(
        (isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "argparse")
        for node in ast.walk(tree)
    )
    if not has_argparse_import:
        return False

    has_main_guard = any(
        isinstance(node, ast.If) and _is_dunder_main_check(node.test)
        for node in tree.body
    )
    return has_main_guard


def _positional_add_argument_names(tree: ast.Module) -> List[str]:
    """Return the literal name strings passed to ``<parser>.add_argument()``
    calls, in source order, for positional arguments only.

    argparse convention: an argument name with no leading dash
    (``add_argument("inputFolder")``) is positional; one with a leading
    dash (``add_argument("--verbose")``) is an optional flag. Only the
    former counts toward the KTC (input, output, category) signature.
    """
    names: List[str] = []

    for node in _walk_in_order(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue

        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            name = first_arg.value
            if not name.startswith("-"):
                names.append(name)

    return names
