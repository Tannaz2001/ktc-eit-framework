"""
src/ktc_framework/adapters/cli_plugin_wrapper.py
---------------------------------------------------
Adapts a raw KTC-competition-contract CLI script (detected as
``plugin_detector.CONTRACT_CLI``) into a ``MethodPlugin`` the in-process
``BatchRunner`` can call like any other reconstruction method.

THE CONTRACT BEING WRAPPED
---------------------------
Every KTC2023 competition entry ships as a standalone script with this
shape::

    python main.py <inputFolder> <outputFolder> <categoryNbr>

It reads, from *inputFolder*:
    ref.mat   — fields Injref, Uelref, Mpat   (empty-tank reference data)
    data*.mat — fields Inj,    Uel,    Mpat   (object measurement data)

It writes, into *outputFolder*:
    1.mat — field 'reconstruction', a (256, 256) array with labels
            {0: background, 1: resistive, 2: conductive}.

It typically imports sibling helper modules (KTCFwd, KTCMeshing,
KTCRegularization, KTCScoring, KTCAux) and loads ``Mesh_sparse.mat`` using a
bare relative filename — both of which only resolve correctly if the
process's *current working directory* is the script's own directory and
those files are either physically present there or importable via
sys.path.

WHAT THIS WRAPPER DOES
------------------------
1. Serialises the in-memory ``DataBatch`` fields to ``ref.mat`` /
   ``data1.mat`` in a temp input directory, using the exact field names
   above.
2. Makes sure the script's helper modules + mesh file are reachable (copies
   any missing ones from ``scoring_path`` next to the script, and also
   prepends ``scoring_path`` to ``PYTHONPATH`` as a second line of defence
   for the Python-import half of that problem — see
   ``_copy_missing_helpers`` for why copying is unavoidable for the
   Mesh_sparse.mat half).
3. Runs the script as a subprocess with ``cwd`` set to the script's own
   directory.
4. Reads back the output ``.mat``, validates shape/labels, and returns a
   ``(256, 256)`` uint8 array.

Every failure mode (bad exit code, missing output, malformed shape, a
timeout, anything) is logged and degrades to an all-zero array rather than
raising — a single mis-behaving upload must never abort a benchmark run
across all methods/levels/samples.

FIELD-MAPPING ASSUMPTION (read this before trusting scores)
-------------------------------------------------------------
``DataBatch`` does not carry separate "reference" and "object" injection/
measurement patterns — only one ``injection_patterns`` and one
``measurement_patterns``. The KTC dataset uses the same injection and
measurement pattern for both the reference (empty tank) and object
measurements within a given difficulty category, so this wrapper reuses
``batch.injection_patterns`` / ``batch.measurement_patterns`` for both
``ref.mat`` and ``data1.mat``. If a specific submission's evaluation data
used different patterns for the two, this mapping under-represents that
and would need ``DataBatch`` extended with a separate reference pattern
field — that plumbing does not exist yet.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.ktc_framework.methods.method_plugin import MethodPlugin
from src.ktc_framework.types import DataBatch

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults — every one of these can be overridden per-instance
# via CLIScriptPlugin's constructor arguments.
# ---------------------------------------------------------------------------

# Env var a power user can set to point at wherever the KTC scoring/helper
# modules live, without touching code.
_SCORING_PATH_ENV = "KTC_SCORING_PATH"

# Env var to force a specific Python interpreter (e.g. a conda env with the
# exact numpy/scipy versions a given submission was built against).
_SOLVER_ENV_VAR = "KTC_SOLVER_ENV"

# Env var for the shared FEM mesh file. Kept separate from scoring_path
# because in this project's actual layout the two live in different
# directories, AND there are two *incompatible formats* of Mesh_sparse.mat
# in this repo:
#   - Codes_Matlab/Mesh_sparse.mat (and configs/ktc_all_methods.yaml's own
#     `mesh_path:`) is a nested {Mesh, Mesh2} struct — this framework's own
#     convention, read by BatchRunner._load_mesh for the built-in methods.
#   - external_methods/abc1/Mesh_sparse.mat is FLAT (top-level g, H,
#     elfaces, Element, ...) — the genuine KTC competition format every raw
#     main.py-style CLI submission expects via a bare
#     `scipy.io.loadmat('Mesh_sparse.mat')['g']`.
# Pointing a CLI script at the nested-format file fails with
# `KeyError: 'g'` (caught by CLIScriptPlugin, degrading to a silent
# all-zero reconstruction) — so the default here MUST be the flat one.
_MESH_PATH_ENV = "KTC_MESH_PATH"

_DEFAULT_SCORING_PATH = "data/KTCScoring"
_DEFAULT_MESH_PATH = "external_methods/abc1/Mesh_sparse.mat"
_DEFAULT_TIMEOUT_S = 180


def derive_cli_method_name(stem: str, existing: Optional[set] = None) -> str:
    """Turn a script's filename stem into a valid registry key.

    Raw KTC CLI submissions have no author-declared name the way
    method.yaml bundles do (``manifest.name``) — every competition entry
    is conventionally just "main.py" — so the registry key is derived
    from the uploaded filename instead.

    This is called from two places that must agree on the exact same name
    for the exact same file:

    1. ``app.py``'s upload handler, at upload time — the returned name is
       what gets written into the YAML config's ``methods:`` list.
    2. ``registry.load_cli_scripts``, inside every fresh benchmark
       subprocess — it re-discovers the same file on disk and must derive
       the identical name so ``methods: [<name>]`` in the config actually
       resolves to this wrapper, instead of failing with "not registered".

    Parameters
    ----------
    stem:
        The script's filename without extension (e.g. ``"main"`` for
        ``main.py``).
    existing:
        When given, a numeric suffix is appended on collision (e.g. a
        second upload also named ``main.py`` becomes ``"main_2"`` —
        this is what the dashboard upload path wants, since two
        different files can legitimately need distinct names).
        When ``None`` (the default), no suffix is added — this is what
        subprocess-side rediscovery of a single already-on-disk file
        wants: reproduce the one name that was already assigned, not
        invent a new one.
    """
    base = re.sub(r"\W+", "_", stem).strip("_") or "CLIMethod"
    if not base[0].isalpha():
        base = f"CLI_{base}"

    if existing is None:
        return base

    name = base
    suffix = 2
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def create_cli_wrapper_class(
    script_path: str,
    name: str,
    scoring_path: Optional[str] = None,
    mesh_path: Optional[str] = None,
    python_exec: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> type:
    """Build a zero-argument-constructible ``CLIScriptPlugin`` subclass.

    The method registry (``src.ktc_framework.registry``) stores classes and
    instantiates them with no arguments — ``registry_get(name)()`` — the
    same way it instantiates every other method. ``CLIScriptPlugin`` itself
    needs a ``script_path`` at construction time, so this factory closes
    over the config and returns a subclass whose ``__init__`` takes
    nothing, following the exact pattern
    ``subprocess_wrapper.create_wrapper_class`` uses for ``method.yaml``
    bundles.

    Parameters
    ----------
    script_path, scoring_path, mesh_path, python_exec, timeout:
        Forwarded to ``CLIScriptPlugin.__init__`` — see its docstring.
    name:
        The class name (and therefore registry key) the wrapper is
        registered under. Callers are responsible for choosing something
        unique, since raw CLI submissions have no author-declared name the
        way ``method.yaml`` bundles do (see ``app.py``'s
        ``_register_cli_script`` for the naming/collision logic).

    Returns
    -------
    type
        A ``CLIScriptPlugin`` subclass named *name*, ready to pass to
        ``register_method``.
    """
    _script_path = script_path
    _scoring_path = scoring_path
    _mesh_path = mesh_path
    _python_exec = python_exec
    _timeout = timeout

    class _CLIWrapper(CLIScriptPlugin):
        def __init__(self) -> None:
            super().__init__(
                script_path=_script_path,
                scoring_path=_scoring_path,
                mesh_path=_mesh_path,
                python_exec=_python_exec,
                timeout=_timeout,
            )

    _CLIWrapper.__name__ = name
    _CLIWrapper.__qualname__ = name
    return _CLIWrapper


class CLIScriptPlugin(MethodPlugin):
    """Wraps one KTC-contract CLI script as an in-process MethodPlugin.

    Parameters
    ----------
    script_path:
        Absolute or relative path to the uploaded ``main.py``-style script.
    scoring_path:
        Directory containing the shared KTC ``.py`` helper modules
        (KTCFwd, KTCMeshing, KTCRegularization, KTCScoring, KTCAux,
        KTCPlotting, ...). Every ``.py`` file found here is copied next to
        the script if not already present — see ``_copy_missing_helpers``.
        Defaults to ``$KTC_SCORING_PATH`` if set, else ``'data/KTCScoring'``.
    mesh_path:
        Path to the shared ``Mesh_sparse.mat``. Kept as a separate
        parameter from ``scoring_path`` because the two commonly live in
        different directories (in this project: ``data/KTCScoring/`` for
        the helpers, ``Codes_Matlab/Mesh_sparse.mat`` for the mesh).
        Defaults to ``$KTC_MESH_PATH`` if set, else
        ``'Codes_Matlab/Mesh_sparse.mat'``.
    python_exec:
        Python interpreter to run the script with. Defaults to
        ``$KTC_SOLVER_ENV`` if set, else ``sys.executable``. Use this when
        a submission needs package versions that conflict with the main
        framework's environment.
    timeout:
        Hard wall-clock limit (seconds) for one reconstruct() call.
    """

    def __init__(
        self,
        script_path: str,
        scoring_path: Optional[str] = None,
        mesh_path: Optional[str] = None,
        python_exec: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.script_path = Path(script_path).resolve()
        if not self.script_path.exists():
            # Fail loud at construction time — a missing script is a
            # configuration error, not a per-sample runtime condition.
            raise FileNotFoundError(f"CLIScriptPlugin: script not found: {self.script_path}")

        self.scoring_path = Path(
            scoring_path or os.environ.get(_SCORING_PATH_ENV, _DEFAULT_SCORING_PATH)
        ).resolve()
        self.mesh_path = Path(
            mesh_path or os.environ.get(_MESH_PATH_ENV, _DEFAULT_MESH_PATH)
        ).resolve()

        self.python_exec = python_exec or os.environ.get(_SOLVER_ENV_VAR) or sys.executable
        self.timeout = timeout

    # ------------------------------------------------------------------
    # MethodPlugin interface
    # ------------------------------------------------------------------

    def reconstruct(self, batch: DataBatch) -> np.ndarray:
        tmp_in: Optional[str] = None
        tmp_out: Optional[str] = None

        try:
            script_dir = self.script_path.parent

            # Best-effort: place the shared helper modules + mesh file next
            # to the script if they aren't already there.
            _copy_missing_helpers(self.scoring_path, self.mesh_path, script_dir)

            tmp_in = tempfile.mkdtemp(prefix="cliplugin_in_")
            tmp_out = tempfile.mkdtemp(prefix="cliplugin_out_")

            self._write_inputs(batch, Path(tmp_in))

            env = _build_subprocess_env(self.scoring_path)
            cmd = [
                self.python_exec,
                str(self.script_path),
                tmp_in,
                tmp_out,
                str(int(batch.level)),
            ]

            _logger.info(
                "CLIScriptPlugin: running %s — level=%s sample=%s",
                self.script_path.name, batch.level, getattr(batch, "sample_id", "?"),
            )

            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                _logger.warning(
                    "CLIScriptPlugin: %s exited %d.\n  stdout: %s\n  stderr: %s",
                    self.script_path.name,
                    result.returncode,
                    (result.stdout or "")[-2000:],
                    (result.stderr or "")[-2000:],
                )
                return np.zeros((256, 256), dtype=np.uint8)

            out_mat = _find_output_mat(Path(tmp_out))
            if out_mat is None:
                _logger.warning(
                    "CLIScriptPlugin: %s produced no .mat file in '%s'.\n  stderr: %s",
                    self.script_path.name, tmp_out, (result.stderr or "")[-2000:],
                )
                return np.zeros((256, 256), dtype=np.uint8)

            import scipy.io  # local import — keeps the top-level import list minimal

            mat = scipy.io.loadmat(str(out_mat), squeeze_me=True, struct_as_record=False)

            if "reconstruction" not in mat:
                _logger.warning(
                    "CLIScriptPlugin: %s output has no 'reconstruction' key. Keys: %s",
                    self.script_path.name,
                    [k for k in mat if not k.startswith("_")],
                )
                return np.zeros((256, 256), dtype=np.uint8)

            reconstruction = np.asarray(mat["reconstruction"]).astype(np.uint8)

            if reconstruction.shape != (256, 256):
                _logger.warning(
                    "CLIScriptPlugin: %s reconstruction shape %s != (256, 256).",
                    self.script_path.name, reconstruction.shape,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            if not np.all(np.isin(reconstruction, [0, 1, 2])):
                _logger.warning(
                    "CLIScriptPlugin: %s reconstruction has labels outside {0,1,2}.",
                    self.script_path.name,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            # Defence-in-depth — matches the MethodPlugin base-class contract.
            # Shape/label checks above already make this a no-op in practice.
            self.validate_output(reconstruction)
            return reconstruction

        except subprocess.TimeoutExpired:
            _logger.warning(
                "CLIScriptPlugin: %s timed out after %ds.",
                self.script_path.name, self.timeout,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        except Exception as exc:  # noqa: BLE001 — any failure degrades to zeros
            _logger.warning(
                "CLIScriptPlugin: %s unexpected error — %s: %s",
                self.script_path.name, type(exc).__name__, exc,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        finally:
            for tmp_dir in (tmp_in, tmp_out):
                if tmp_dir and os.path.isdir(tmp_dir):
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_inputs(self, batch: DataBatch, in_dir: Path) -> None:
        """Serialise *batch* into ``ref.mat`` and ``data1.mat`` inside
        *in_dir*, using the exact field names the KTC contract expects.

        See the module docstring's "FIELD-MAPPING ASSUMPTION" section for
        why ``injection_patterns`` / ``measurement_patterns`` are reused
        for both files.
        """
        import scipy.io

        inj = np.asarray(batch.injection_patterns, dtype=np.float64)
        # Column vector, not flat 1-D: scipy.io.savemat serialises a plain
        # (N,) array as a (1, N) MATLAB row vector, but the KTC scripts
        # index these with a length-N boolean mask along axis 0
        # (`deltaU[vincl]`), which requires a (N, 1) column vector —
        # matching the shape real KTC-provided .mat files use. Without
        # this reshape the script crashes with "boolean index did not
        # match indexed array along axis 0; size of axis is 1".
        uel = np.asarray(batch.voltages, dtype=np.float64).reshape(-1, 1)

        mpat = (
            np.asarray(batch.measurement_patterns, dtype=np.float64)
            if batch.measurement_patterns is not None
            else np.zeros((31, 32), dtype=np.float64)
        )

        if batch.reference_voltages is not None:
            uelref = np.asarray(batch.reference_voltages, dtype=np.float64).reshape(-1, 1)
        else:
            # No ref.mat was found by the data loader upstream. Writing
            # zeros lets the script run instead of crashing, but the
            # reconstruction will likely be poor — this is surfaced as a
            # warning rather than silently swallowed.
            uelref = np.zeros_like(uel)
            _logger.warning(
                "CLIScriptPlugin: batch has no reference_voltages — writing "
                "zeros for Uelref. Check that ref.mat was found by the "
                "data loader; reconstruction quality will likely suffer."
            )

        scipy.io.savemat(str(in_dir / "ref.mat"), {
            "Injref": inj,
            "Uelref": uelref,
            "Mpat": mpat,
        })
        scipy.io.savemat(str(in_dir / "data1.mat"), {
            "Inj": inj,
            "Uel": uel,
            "Mpat": mpat,
        })


# ---------------------------------------------------------------------------
# Module-level helpers — standalone so they're easy to unit-test in
# isolation from the class and the subprocess machinery.
# ---------------------------------------------------------------------------

def _copy_missing_helpers(scoring_path: Path, mesh_path: Path, script_dir: Path) -> None:
    """Copy the shared KTC ``.py`` helper modules and the mesh file into
    *script_dir* if not already present there. Never overwrites an
    existing file — an uploaded bundle may ship its own version
    deliberately.

    Helper modules are discovered by globbing ``*.py`` in *scoring_path*
    rather than a fixed filename list — different submissions import
    different subsets (e.g. one script needs KTCPlotting, another doesn't),
    and a hardcoded list silently breaks the next script that needs a file
    nobody thought to add. Globbing means any new helper module dropped
    into *scoring_path* is picked up automatically.

    The mesh file is handled separately, via its own *mesh_path*, because
    in this project's actual layout it does not live next to the ``.py``
    helpers (helpers are under ``data/KTCScoring/``, the mesh is under
    ``Codes_Matlab/``) — see the ``mesh_path`` parameter docs on
    ``CLIScriptPlugin`` for why a single shared root doesn't work.

    Why copy instead of relying only on PYTHONPATH
    -----------------------------------------------
    KTC scripts load ``Mesh_sparse.mat`` with a bare relative filename
    (e.g. ``scipy.io.loadmat('Mesh_sparse.mat')``). That resolves against
    the process's current working directory, not ``sys.path`` — so
    PYTHONPATH (see ``_build_subprocess_env``) cannot fix this half of the
    problem. Physically placing the file next to the script is the only
    reliable fix. The ``.py`` helper modules technically could be found via
    PYTHONPATH alone, but copying them too means the script behaves
    identically to how it would in its original repo layout.

    Why copy instead of symlink
    -----------------------------
    Creating a symlink on Windows normally requires Developer Mode or
    admin rights; a plain file copy works everywhere with no privilege
    requirements.
    """
    if not scoring_path.is_dir():
        _logger.warning(
            "CLIScriptPlugin: scoring_path '%s' does not exist — cannot "
            "supply KTC helper modules.", scoring_path,
        )
    else:
        for src in scoring_path.glob("*.py"):
            dst = script_dir / src.name
            if not dst.exists():
                try:
                    shutil.copy2(str(src), str(dst))
                except OSError as exc:
                    _logger.warning(
                        "CLIScriptPlugin: could not copy %s into %s: %s",
                        src.name, script_dir, exc,
                    )

    dst_mesh = script_dir / "Mesh_sparse.mat"
    if not mesh_path.exists():
        _logger.warning(
            "CLIScriptPlugin: mesh_path '%s' does not exist — script will "
            "likely fail when it loads Mesh_sparse.mat.", mesh_path,
        )
    elif not dst_mesh.exists() or not _has_flat_mesh_keys(dst_mesh):
        # A file may already sit here in the wrong (nested {Mesh, Mesh2})
        # format — e.g. copied for an unrelated purpose, or left over from
        # this project's own convention (see the module-level comment on
        # _DEFAULT_MESH_PATH). "Never overwrite" is meant to protect a
        # deliberately-placed compatible file, not preserve a stale
        # incompatible one that would make every reconstruction silently
        # degrade to zeros via a caught KeyError.
        try:
            shutil.copy2(str(mesh_path), str(dst_mesh))
        except OSError as exc:
            _logger.warning(
                "CLIScriptPlugin: could not copy Mesh_sparse.mat into %s: %s",
                script_dir, exc,
            )


def _has_flat_mesh_keys(mat_path: Path) -> bool:
    """True if *mat_path* has the flat top-level ``'g'`` key raw KTC CLI
    scripts read directly (``mat_dict_mesh['g']``), as opposed to this
    framework's own nested ``{Mesh, Mesh2}`` struct convention.

    Uses ``scipy.io.whosmat`` — reads only variable name/shape metadata,
    not the full arrays, so checking this on every ``reconstruct()`` call
    is cheap.
    """
    import scipy.io

    try:
        names = {entry[0] for entry in scipy.io.whosmat(str(mat_path))}
    except Exception:
        return False
    return "g" in names


def _build_subprocess_env(scoring_path: Path) -> dict:
    """Return a copy of the current environment with *scoring_path*
    prepended to PYTHONPATH.

    Backup mechanism for ``import KTCFwd`` / ``import KTCScoring`` etc. in
    case ``_copy_missing_helpers`` could not physically place the .py
    files next to the script (e.g. a read-only script directory).
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{scoring_path}{os.pathsep}{existing}" if existing else str(scoring_path)
    )
    return env


def _find_output_mat(out_dir: Path) -> Optional[Path]:
    """Locate the script's output ``.mat`` file inside *out_dir*.

    The KTC contract specifies ``1.mat`` exactly, but not every submission
    follows that literally, so this falls back to the first ``.mat`` file
    found if ``1.mat`` isn't there.
    """
    canonical = out_dir / "1.mat"
    if canonical.exists():
        return canonical

    candidates = sorted(out_dir.glob("*.mat"))
    return candidates[0] if candidates else None
