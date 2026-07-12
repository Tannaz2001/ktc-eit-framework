"""
src/ktc_framework/runner/env_resolver.py
-----------------------------------------
Dynamic conda-environment discovery and dependency matching.

PURPOSE
-------
ML-based reconstruction methods (TensorFlow, PyTorch, FEniCS …) often require
a specific conda environment because their heavy dependencies conflict with each
other in a single interpreter.  This module automates answering the question:

    "Given a third-party repo, which conda env should we activate to run it?"

THREE PUBLIC FUNCTIONS
----------------------
build_env_index   – probe every conda env; record which packages are importable.
parse_repo_imports – read a repo's dependency file; return normalised import names.
resolve_env       – match those names to an env from the index.

DEPENDENCIES
------------
stdlib only, plus PyYAML for environment.yml parsing (conda ships it;
install separately: pip install pyyaml).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# PyYAML — optional at import time; only required when parsing environment.yml.
# Deferring the import here means the rest of the module works without it.
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Candidate package set
# ---------------------------------------------------------------------------
# These are the only packages we probe for inside conda environments.
# Chosen because they are:
#   • commonly needed by competition / research EIT repos
#   • non-trivial to install (environment-specific, not always in base)
# Add new entries here as the ecosystem grows; the probe script picks them up
# automatically — no other changes needed.
_PROBE_CANDIDATES: List[str] = [
    "numpy",
    "scipy",
    "skimage",         # scikit-image
    "tensorflow",
    "torch",           # PyTorch
    "torch_geometric", # PyTorch Geometric
    "dolfin",          # FEniCS (legacy, 2019.x)
    "dolfinx",         # FEniCSx (new API, distinct package from dolfin)
    "deepinv",         # DeepInverse
    "cv2",             # opencv-python
]

# ---------------------------------------------------------------------------
# Probe script
# ---------------------------------------------------------------------------
# This string is passed verbatim to `python -c "..."` inside each conda env.
# It uses importlib.util.find_spec — fast, no side-effects, does not actually
# *load* TensorFlow etc. — and prints a single JSON line to stdout.
#
# Each candidate is tried independently so one broken package cannot silence
# all the others (e.g. a partially installed dolfin won't hide numpy).
#
# The {cands!r} placeholder is filled at module load time by .format() below.
# Double braces {{ }} become literal { } after .format() is applied.
_PROBE_SCRIPT: str = (
    "import platform, json, importlib.util\n"
    "cands = {cands!r}\n"
    "ok = []\n"
    "for c in cands:\n"
    "    try:\n"
    "        if importlib.util.find_spec(c) is not None:\n"
    "            ok.append(c)\n"
    "    except Exception:\n"
    "        pass\n"
    "print(json.dumps({{'python': platform.python_version(), 'imports': ok}}))\n"
).format(cands=_PROBE_CANDIDATES)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class EnvError(Exception):
    """Raised when no conda environment satisfies a method's import requirements.

    The exception message includes:
      • the full set of packages that were needed, and
      • per environment, which packages were missing.

    Example message
    ---------------
    No conda env satisfies {'tensorflow', 'torch_geometric'}.
      Per-environment gaps:
        base                : missing {'tensorflow', 'torch_geometric'}
        scipy-env           : missing {'tensorflow', 'torch_geometric'}
        tf-env              : missing {'torch_geometric'}
    """


# ---------------------------------------------------------------------------
# Internal helpers — not part of the public API
# ---------------------------------------------------------------------------

def _conda_executable() -> str:
    """Return the absolute path to the conda executable.

    Search order:
      1. $CONDA_EXE environment variable — set automatically when a conda
         base environment is active; most reliable on all platforms.
      2. 'conda' resolved via PATH — works when conda is on PATH but no
         environment is active (e.g. in a CI container).

    Raises
    ------
    EnvironmentError
        If conda cannot be located by either method.
    """
    # $CONDA_EXE is set by conda's shell integration; prefer it.
    exe = os.environ.get("CONDA_EXE", "")
    if exe and Path(exe).exists():
        return exe

    # Fall back to PATH resolution (handles conda.bat on Windows too).
    on_path = shutil.which("conda")
    if on_path:
        return on_path

    raise EnvironmentError(
        "conda not found.  "
        "Either activate the base conda environment (so $CONDA_EXE is set) "
        "or ensure 'conda' is on $PATH."
    )


def _env_name_from_path(env_path: str) -> str:
    """Derive a human-readable name from a conda environment's filesystem path.

    Standard conda layout::

        <conda_root>/               ← base environment
        <conda_root>/envs/<name>/   ← named environment

    Named envs always have 'envs' as their parent directory.  Everything else
    is considered the base environment.

    Note: if two separate conda installations both have an env called 'myenv',
    the second one will silently overwrite the first in the index dict.  This
    is an accepted limitation for typical single-conda-root setups.

    Parameters
    ----------
    env_path:
        Absolute path returned by ``conda env list --json``.

    Returns
    -------
    str
        E.g. ``'base'``, ``'tf-env'``, ``'pyeit'``.
    """
    p = Path(env_path)
    # Named envs live under <root>/envs/<name>  ⟹  parent.name == "envs"
    if p.parent.name == "envs":
        return p.name
    # Anything else (the root itself, a custom prefix, etc.) is 'base'.
    return "base"


def _probe_env(conda: str, env_path: str) -> dict:
    """Run the probe script inside a single conda environment.

    Uses ``conda run -p <path>`` (path-based activation) so the function works
    regardless of whether the env has a registered name in conda's metadata.

    Parameters
    ----------
    conda:
        Absolute path to the conda executable.
    env_path:
        Absolute path to the environment directory.

    Returns
    -------
    dict
        On success: ``{'python': '3.10.12', 'imports': ['numpy', 'tensorflow']}``
        On any failure: ``{'python': None, 'imports': []}``
    """
    _FAILURE: dict = {"python": None, "imports": []}

    try:
        result = subprocess.run(
            # -p <path> activates by path, not by name — more robust.
            # python -c runs our probe script in that env's interpreter.
            [conda, "run", "-p", env_path, "python", "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=60,  # TF importlib probe should finish well under 10 s
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Timeout or conda itself missing — record as failure and move on.
        return _FAILURE

    if result.returncode != 0:
        return _FAILURE

    # The probe prints exactly one JSON line.  conda may prepend preamble text
    # (e.g. activation messages), so scan backwards for the first JSON line.
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # keep scanning upward

    return _FAILURE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_env_index(
    cache_path: str = ".env_index.json",
    force: bool = False,
) -> dict:
    """Discover all conda environments and record their importable packages.

    Results are cached on disk so repeated calls don't re-probe every env.
    After installing new packages into a conda env, pass ``force=True`` to
    rebuild the cache.

    Parameters
    ----------
    cache_path:
        Path to the JSON cache file.
        Default: ``.env_index.json`` in the current working directory.
    force:
        If ``True``, ignore any existing cache and re-probe everything.

    Returns
    -------
    dict
        Schema::

            {
              "<env_name>": {
                "python":  "<version>" | null,
                "imports": ["numpy", "tensorflow", ...],
                "path":    "/abs/path/to/env"
              },
              ...
            }

        ``python`` is ``null`` / ``None`` when the probe subprocess failed
        (e.g. corrupted env, missing Python binary).

    Raises
    ------
    EnvironmentError
        If conda is not found or ``conda env list`` fails.
    """
    # ── 1. Return cached result when available ───────────────────────────
    cache = Path(cache_path)
    if cache.exists() and not force:
        with cache.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ── 2. List all conda environments ───────────────────────────────────
    conda = _conda_executable()

    try:
        list_proc = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        raise EnvironmentError(
            f"Failed to run 'conda env list --json': {exc}"
        ) from exc

    if list_proc.returncode != 0:
        raise EnvironmentError(
            f"'conda env list --json' exited {list_proc.returncode}:\n"
            f"{list_proc.stderr}"
        )

    # Output is {"envs": ["/path/to/base", "/path/to/named-env", ...]}
    env_paths: List[str] = json.loads(list_proc.stdout).get("envs", [])

    # ── 3. Probe each environment for importable packages ────────────────
    index: dict = {}
    for env_path in env_paths:
        name = _env_name_from_path(env_path)
        probe = _probe_env(conda, env_path)
        # Store both capability data and the raw path so callers can use
        # `conda run -p <path>` directly if they prefer path-based activation.
        index[name] = {
            "python":  probe.get("python"),
            "imports": probe.get("imports", []),
            "path":    env_path,
        }

    # ── 4. Persist cache and return ───────────────────────────────────────
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    return index


def parse_repo_imports(repo_path: str) -> Set[str]:
    """Extract the set of importable package names a repo depends on.

    Reads the first dependency file found in ``repo_path`` (priority order):
      1. ``environment.yml``
      2. ``environment.yaml``
      3. ``requirements.txt``

    Package names are normalised (version pins stripped, aliases resolved) and
    then intersected with ``_PROBE_CANDIDATES`` — packages outside that set
    are silently ignored because we cannot probe for them.

    Alias mapping (distribution name → import name)
    -----------------------------------------------
    scikit-image / scikit_image  →  skimage
    opencv-python / opencv       →  cv2
    fenics / fenics_dolfin       →  dolfin
    torch-geometric              →  torch_geometric
    tensorflow-gpu               →  tensorflow

    Parameters
    ----------
    repo_path:
        Path to the root directory of the external method repository.

    Returns
    -------
    set[str]
        Normalised import names, e.g. ``{'tensorflow', 'numpy', 'cv2'}``.
        Empty set if no dependency file is found.

    Raises
    ------
    ImportError
        If an ``environment.yml`` is found but PyYAML is not installed.
    """
    # Map distribution/conda package names to the importable module name.
    # Only entries where the two names *differ* are listed here; everything
    # else falls through to the raw name.
    _ALIAS: Dict[str, str] = {
        "scikit-image":    "skimage",
        "scikit_image":    "skimage",
        "opencv-python":   "cv2",
        "opencv_python":   "cv2",
        "opencv":          "cv2",
        "fenics":          "dolfin",
        "fenics_dolfin":   "dolfin",
        "fenicsx":         "dolfinx",
        "fenics-dolfinx":  "dolfinx",
        "fenics_dolfinx":  "dolfinx",
        "torch-geometric": "torch_geometric",
        "tensorflow-gpu":  "tensorflow",
        "tensorflow_gpu":  "tensorflow",
    }

    root = Path(repo_path)
    raw_names: List[str] = []

    # ── Locate the first available dependency file ─────────────────────
    yml_file: Optional[Path] = next(
        (root / name for name in ("environment.yml", "environment.yaml")
         if (root / name).exists()),
        None,
    )
    req_file: Optional[Path] = root / "requirements.txt"

    if yml_file is not None:
        # ── Parse environment.yml ────────────────────────────────────────
        # Requires PyYAML.
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to parse environment.yml.  "
                "Install it:  pip install pyyaml"
            )
        with yml_file.open("r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)

        # environment.yml structure:
        #   dependencies:
        #     - numpy
        #     - conda-forge::scipy==1.11
        #     - pip:
        #         - tensorflow>=2.10
        if isinstance(data, dict):
            for entry in data.get("dependencies", []):
                if isinstance(entry, str):
                    # Strip conda channel prefix (e.g. "conda-forge::numpy")
                    pkg = entry.split("::")[-1]
                    raw_names.append(_strip_version(pkg))
                elif isinstance(entry, dict):
                    # pip sub-block: {"pip": ["tensorflow>=2", ...]}
                    for pip_pkg in entry.get("pip", []):
                        if isinstance(pip_pkg, str):
                            raw_names.append(_strip_version(pip_pkg))

    elif req_file.exists():
        # ── Parse requirements.txt ────────────────────────────────────────
        with req_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                # Skip blank lines, comments, and flag lines (-r, -c, -e, --index-url …)
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Strip inline comments then version pins
                pkg = line.split("#")[0].strip()
                raw_names.append(_strip_version(pkg))

    # ── Normalise names and intersect with probe set ──────────────────
    probe_set: Set[str] = set(_PROBE_CANDIDATES)
    normalised: Set[str] = set()

    for name in raw_names:
        # Resolve alias: try exact match first, then lowercase.
        resolved = _ALIAS.get(name) or _ALIAS.get(name.lower()) or name
        # Only keep packages the probe can actually test for.
        if resolved in probe_set:
            normalised.add(resolved)

    return normalised


def _strip_version(pkg: str) -> str:
    """Remove version specifier from a package name string.

    Handles all common PEP 508 / conda pin operators:
    ``==``, ``>=``, ``<=``, ``!=``, ``~=``, ``>``, ``<``, ``=`` (conda).

    Examples
    --------
    >>> _strip_version("tensorflow>=2.10")
    'tensorflow'
    >>> _strip_version("scipy=1.11.0")
    'scipy'
    """
    # Ordered longest-first so ">=" is matched before ">"
    for sep in ("==", ">=", "<=", "!=", "~=", "=", ">", "<"):
        if sep in pkg:
            pkg = pkg.split(sep)[0]
            break
    return pkg.strip()


def resolve_env(
    repo_path: str,
    env_index: Optional[dict] = None,
) -> Optional[str]:
    """Find the conda environment whose imports satisfy a repo's requirements.

    Parameters
    ----------
    repo_path:
        Path to the root of the external method repository.
    env_index:
        Pre-built environment index (from :func:`build_env_index`).
        When ``None``, ``build_env_index()`` is called automatically using
        the default cache path.

    Returns
    -------
    str or None
        Name of the matching conda environment (e.g. ``'tf-env'``), or
        ``None`` when all required packages are "lightweight" (numpy / scipy /
        skimage only) — in that case the caller should use ``sys.executable``
        directly rather than spinning up a conda subprocess.

    Raises
    ------
    EnvError
        When the required packages cannot be satisfied by any environment.
        The message shows what is needed and what each env is missing.
    """
    if env_index is None:
        env_index = build_env_index()

    needed: Set[str] = parse_repo_imports(repo_path)

    # ── Special case: lightweight repo ───────────────────────────────────
    # numpy, scipy, and skimage are installed in virtually every scientific
    # Python environment.  Running such a repo in a conda subprocess adds
    # overhead with no benefit — signal to the caller to use sys.executable.
    _LIGHTWEIGHT: Set[str] = {"numpy", "scipy", "skimage"}
    if needed <= _LIGHTWEIGHT:
        return None  # caller: use sys.executable, no conda activation needed

    # ── Find first environment that provides all needed packages ─────────
    for env_name, caps in env_index.items():
        available: Set[str] = set(caps.get("imports", []))
        if needed <= available:
            # Every required import is available here — use this env.
            return env_name

    # ── No match: build a diagnostic error message ────────────────────────
    lines = [f"No conda env satisfies {needed!r}.", "  Per-environment gaps:"]
    for env_name, caps in env_index.items():
        available = set(caps.get("imports", []))
        missing   = needed - available
        if missing:
            lines.append(f"    {env_name:<22s}: missing {missing!r}")
        else:
            # Theoretically should have matched above — suggest cache rebuild.
            lines.append(
                f"    {env_name:<22s}: (index may be stale — "
                f"re-run build_env_index(force=True))"
            )

    raise EnvError("\n".join(lines))



def resolve(manifest) -> Optional[str]:
    """Resolve a python interpreter path for a MethodManifest's bundle.

    Thin additive wrapper around ``resolve_env()``/``build_env_index()`` for
    callers (e.g. the manifest-driven subprocess wrapper) that need an actual
    interpreter path to put in a subprocess command, not just a conda env
    name. Best-effort: returns ``None`` (never raises) whenever conda isn't
    available, no environment matches, or the repo only needs lightweight
    packages — callers should fall back to their own interpreter discovery
    in all of those cases (e.g. ``env_resolver.resolve(manifest) or python``).

    Parameters
    ----------
    manifest:
        A ``MethodManifest`` (or any object with a ``bundle_dir`` attribute
        pointing at the extracted method's directory).

    Returns
    -------
    str or None
        Absolute path to a python executable inside the matched conda env,
        or ``None`` if nothing could be resolved.
    """
    try:
        env_index = build_env_index()
        env_name = resolve_env(str(manifest.bundle_dir), env_index)
    except Exception:
        return None

    if env_name is None or env_name not in env_index:
        return None

    env_dir = Path(env_index[env_name]["path"])
    python_path = env_dir / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"
    return str(python_path) if python_path.exists() else None
