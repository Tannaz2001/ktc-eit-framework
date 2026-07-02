"""Auto-generate a MethodPlugin subclass from a method.yaml manifest.

Replaces the manual subprocess glue code that competition_cnn.py uses.
Given a MethodManifest, ``create_wrapper_class()`` returns a class that:
  - finds a Python interpreter with the required package
  - serializes DataBatch data to temp .mat files
  - invokes the solver as a subprocess
  - parses the output .mat and returns a (256, 256) uint8 array
  - caches results and cleans up temp dirs
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.ktc_framework.methods._opcache import load as cache_load, save as cache_save
from src.ktc_framework.methods.manifest_loader import MethodManifest
from src.ktc_framework.methods.method_plugin import MethodPlugin

_logger = logging.getLogger(__name__)

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Interpreter discovery (generalized from competition_cnn.py)
# ---------------------------------------------------------------------------

def _has_package(python: str, package: str) -> bool:
    try:
        result = subprocess.run(
            [python, "-c",
             f"import importlib.util, sys;"
             f"sys.exit(0 if importlib.util.find_spec('{package}') else 1)"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _find_python_for_package(
    versions: list[str],
    package: str | None,
    env_override: str | None,
) -> str:
    if env_override and env_override in os.environ:
        candidate = os.environ[env_override]
        if Path(candidate).exists():
            _logger.info("Using interpreter from $%s: %s", env_override, candidate)
            return candidate

    if package is None or _has_package(sys.executable, package):
        return sys.executable

    py_launcher = shutil.which("py")
    if py_launcher:
        for ver in versions:
            try:
                result = subprocess.run(
                    [py_launcher, f"-{ver}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    candidate = result.stdout.strip()
                    if _has_package(candidate, package):
                        _logger.info("Found %s on Python %s: %s", package, ver, candidate)
                        return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue

    _logger.warning(
        "No Python with '%s' found. Install it or set $%s. "
        "Subprocess calls will fail.",
        package, env_override or "<ENV_VAR>",
    )
    return sys.executable


# ---------------------------------------------------------------------------
# Data file discovery (generalized from competition_cnn.py)
# ---------------------------------------------------------------------------

def _find_data_file(
    level: int, sample_id: str, sample_map: dict[str, str],
) -> Optional[Path]:
    letter = sample_id
    if "_" in sample_id:
        letter = sample_id.rsplit("_", 1)[-1]
    file_num = sample_map.get(letter, letter)

    # KTC_DATASET_ROOT is set by BatchRunner from the YAML dataset_root field
    env_root = os.environ.get("KTC_DATASET_ROOT", "")
    search_roots = ([Path(env_root)] if env_root else []) + [_FRAMEWORK_ROOT, Path.cwd()]

    # When dataset_root is set, the data is typically directly under it.
    # The richer layout list handles both the Zenodo folder structure and
    # flatter local copies.
    layouts = [
        "evaluation_datasets/level{level}/data{num}.mat",
        "EvaluationData/evaluation_datasets/level{level}/data{num}.mat",
        "EvaluationData/level{level}/data{num}.mat",
        "level{level}/data{num}.mat",
    ]

    for root in search_roots:
        for layout in layouts:
            candidate = root / layout.format(level=level, num=file_num)
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Model fingerprinting (generalized from competition_cnn.py)
# ---------------------------------------------------------------------------

def _compute_model_fingerprint(bundle_dir: Path, weight_files: list[str]) -> str:
    if not weight_files:
        return "noweights"
    try:
        h = hashlib.md5()
        for wf in sorted(weight_files):
            wp = bundle_dir / wf
            if wp.exists():
                h.update(wp.name.encode("utf-8"))
                h.update(str(wp.stat().st_size).encode("utf-8"))
                h.update(str(int(wp.stat().st_mtime)).encode("utf-8"))
        digest = h.hexdigest()[:16]
        return digest if digest else "noweights"
    except Exception:
        return "noweights"


# ---------------------------------------------------------------------------
# Wrapper class factory
# ---------------------------------------------------------------------------

def create_wrapper_class(manifest: MethodManifest) -> type:
    """Dynamically create a MethodPlugin subclass driven by a manifest.

    The returned class has ``__name__ == manifest.name`` so it registers
    under the correct key in the method registry.
    """

    python = _find_python_for_package(
        manifest.python_versions, manifest.check_import, manifest.env_override,
    )
    model_fp = _compute_model_fingerprint(manifest.bundle_dir, manifest.weights)
    entry_point = str(manifest.bundle_dir / manifest.entry_point)
    cwd = str(manifest.bundle_dir / manifest.working_dir)

    class _Wrapper(MethodPlugin):

        def reconstruct(self, batch) -> np.ndarray:
            level = int(batch.level)
            sample_id = str(getattr(batch, "sample_id", "?"))

            cache_key = f"manifest|{manifest.name}|{model_fp}|L{level}|{sample_id}"
            cached = cache_load(cache_key)
            if cached is not None:
                return cached

            data_file = _find_data_file(level, sample_id, manifest.sample_map)
            if data_file is None:
                _logger.error(
                    "%s: data file not found for level=%d sample=%s",
                    manifest.name, level, sample_id,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            tmp_input = None
            tmp_output = None
            try:
                # Pass the actual data directory — no copy needed since the
                # KTC dataset is already on disk in the project.
                input_dir = str(data_file.parent)
                tmp_output = tempfile.mkdtemp(prefix=f"{manifest.name}_out_")

                cmd = [python, entry_point, input_dir, tmp_output, str(level)]

                _logger.info(
                    "%s: subprocess — level=%d sample=%s",
                    manifest.name, level, sample_id,
                )

                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=manifest.timeout,
                )

                if result.returncode != 0:
                    _logger.error(
                        "%s: subprocess exited %d.\n  stdout: %s\n  stderr: %s",
                        manifest.name, result.returncode,
                        (result.stdout or "")[-2000:],
                        (result.stderr or "")[-2000:],
                    )
                    return np.zeros((256, 256), dtype=np.uint8)

                import scipy.io
                output_files = [
                    f for f in os.listdir(tmp_output) if f.endswith(".mat")
                ]
                if not output_files:
                    _logger.error(
                        "%s: no .mat file in output dir '%s'",
                        manifest.name, tmp_output,
                    )
                    return np.zeros((256, 256), dtype=np.uint8)

                out_path = os.path.join(tmp_output, output_files[0])
                mat = scipy.io.loadmat(out_path, squeeze_me=True, struct_as_record=False)

                if "reconstruction" not in mat:
                    _logger.error(
                        "%s: output .mat has no 'reconstruction' key. Keys: %s",
                        manifest.name,
                        [k for k in mat if not k.startswith("_")],
                    )
                    return np.zeros((256, 256), dtype=np.uint8)

                reconstruction = np.asarray(mat["reconstruction"], dtype=np.uint8)

                if reconstruction.shape != (256, 256):
                    _logger.error(
                        "%s: reconstruction shape %s != (256, 256)",
                        manifest.name, reconstruction.shape,
                    )
                    return np.zeros((256, 256), dtype=np.uint8)

                self.validate_output(reconstruction)
                cache_save(cache_key, reconstruction)
                return reconstruction

            except subprocess.TimeoutExpired:
                _logger.error(
                    "%s: subprocess timed out (%ds) for level=%d sample=%s",
                    manifest.name, manifest.timeout, level, sample_id,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            except Exception as exc:
                _logger.error(
                    "%s: unexpected error — %s: %s",
                    manifest.name, type(exc).__name__, exc, exc_info=True,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            finally:
                if tmp_output and os.path.isdir(tmp_output):
                    try:
                        shutil.rmtree(tmp_output)
                    except Exception:
                        pass

    _Wrapper.__name__ = manifest.name
    _Wrapper.__qualname__ = manifest.name
    return _Wrapper
