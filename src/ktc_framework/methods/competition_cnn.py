"""Wraps ABC1 KTC2023 competition entry (Beraldo et al., UFABC). CNN post-processor for EIT segmentation."""

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

from ktc_framework.adapters.method_registry import register
from ktc_framework.methods import _opcache
from ktc_framework.methods.method_plugin import MethodPlugin
from ktc_framework.types import DataBatch


_logger = logging.getLogger(__name__)

# ── Dynamic discovery of ABC1 submission directory ────────────────────────
# Searches up from this file to find the framework root, then looks for
# the ABC1 submission in known locations. Supports git clone, relocation,
# docker mounts, etc. — no hardcoded absolute paths.

def _find_submission_dir() -> Optional[Path]:
    """Locate ABC1 submission directory by searching from framework root.

    Tries (in order):
    1. Relative to framework: ../../../KTC2023-ABC1/KTC2023_Python_A01+
    2. Sibling of framework root: Framework/../KTC2023-ABC1/KTC2023_Python_A01+
    3. Environment variable: $ABC1_SUBMISSION_PATH
    4. CWD variants (for docker/alternate layouts)
    """
    # Framework root is 3 levels up from this file
    # src/ktc_framework/methods/competition_cnn.py -> parents[3] = KTC_WORK_HIS/
    framework_root = Path(__file__).resolve().parents[3]

    candidates = [
        framework_root / "external_methods" / "abc1",             # ← preferred: inside the project
        framework_root.parent / "KTC2023-ABC1" / "KTC2023_Python_A01+",
        framework_root / "KTC2023-ABC1" / "KTC2023_Python_A01+",
        Path(os.environ.get("ABC1_SUBMISSION_PATH", "")),
        Path.cwd() / "KTC2023-ABC1" / "KTC2023_Python_A01+",
    ]

    for path in candidates:
        if path and (path / "main_python.py").exists():
            _logger.info(f"[CompetitionCNN] Found submission at: {path}")
            return path

    _logger.warning(
        f"[CompetitionCNN] Submission directory not found. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates if c)
        + f"\nSet $ABC1_SUBMISSION_PATH or place KTC2023-ABC1 next to framework root."
    )
    return None

def _has_tensorflow(python: str) -> bool:
    """Check whether a Python interpreter has TensorFlow installed.

    Uses importlib.util.find_spec which only checks sys.path without
    actually loading TensorFlow — returns in under a second.
    """
    try:
        result = subprocess.run(
            [python, "-c",
             "import importlib.util, sys;"
             "sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _find_python_interpreter() -> str:
    """Find a Python interpreter with TensorFlow installed.

    Tries (in order):
    1. $ABC1_PYTHON env var — power user override, skips TF check
    2. sys.executable — if TF is already importable here
    3. Windows py launcher scan — py -3.12, -3.11, -3.10 (first with TF wins)
    4. Fallback to sys.executable with a warning
    """
    if "ABC1_PYTHON" in os.environ:
        python_path = os.environ["ABC1_PYTHON"]
        if Path(python_path).exists():
            _logger.info(f"[CompetitionCNN] Using ABC1_PYTHON from env: {python_path}")
            return python_path
        _logger.warning(f"[CompetitionCNN] ABC1_PYTHON points to non-existent: {python_path}")

    if _has_tensorflow(sys.executable):
        _logger.info(f"[CompetitionCNN] TensorFlow found on current Python: {sys.executable}")
        return sys.executable

    import shutil
    py_launcher = shutil.which("py")
    if py_launcher:
        for ver in ["3.12", "3.11", "3.10"]:
            try:
                result = subprocess.run(
                    [py_launcher, f"-{ver}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    candidate = result.stdout.strip()
                    if _has_tensorflow(candidate):
                        _logger.info(f"[CompetitionCNN] TensorFlow found on Python {ver}: {candidate}")
                        return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue

    _logger.warning(
        "[CompetitionCNN] No Python with TensorFlow found. "
        "Install TensorFlow (pip install tensorflow) or set $ABC1_PYTHON. "
        "Subprocess calls will fail."
    )
    return sys.executable

_ABC1_CWD    = _find_submission_dir()
_ABC1_MAIN   = str((_ABC1_CWD / "main_python.py")) if _ABC1_CWD else None
_ABC1_PYTHON = _find_python_interpreter()
_TF_AVAILABLE = _has_tensorflow(_ABC1_PYTHON)


def _model_fingerprint() -> str:
    """Hash of the CNN model weights (.h5), computed once. Included in the
    output-cache key so a changed/retrained model invalidates the cache."""
    if _ABC1_CWD is None:
        return "nomodel"
    try:
        h = hashlib.md5()
        for h5 in sorted(Path(_ABC1_CWD).rglob("*.h5")):
            h.update(h5.name.encode("utf-8"))
            h.update(str(h5.stat().st_size).encode("utf-8"))  # cheap content proxy
            h.update(str(int(h5.stat().st_mtime)).encode("utf-8"))
        return h.hexdigest()[:16] or "nomodel"
    except Exception:
        return "nomodel"


_MODEL_FP = _model_fingerprint()

# ── Import-time diagnostics — visible to every teammate on first run ──────
# These use Rich directly so the warnings appear in the benchmark console
# even when Python's logging module has no handlers configured.
def _warn(msg: str) -> None:
    try:
        from rich.console import Console as _Console
        _Console(safe_box=True).print(msg)
    except ImportError:
        print(msg)

if _ABC1_CWD is None:
    _warn(
        "[bold yellow][CompetitionCNN] ABC1 submission directory not found.[/bold yellow]\n"
        "   Place [bold]KTC2023-ABC1/KTC2023_Python_A01+[/bold] next to the framework root,\n"
        "   or set env var:  [bold]ABC1_SUBMISSION_PATH=/path/to/KTC2023_Python_A01+[/bold]\n"
        "   CNN will return zeros until this is resolved."
    )
elif not _TF_AVAILABLE:
    _warn(
        "[bold yellow][CompetitionCNN] TensorFlow not found on any Python interpreter.[/bold yellow]\n"
        "   Install it:      [bold]pip install tensorflow[/bold]\n"
        "   Or point to a Python that has it:  [bold]ABC1_PYTHON=/path/to/python[/bold]\n"
        "   CNN will return zeros until TensorFlow is available.\n"
        "   Other methods (BackProjection, GaussNewton, etc.) are unaffected."
    )

# Sample-letter to data-file-number mapping (mirrors KTCDataPlugin)
_SAMPLE_MAP: dict[str, str] = {
    "A": "1", "B": "2", "C": "3",
    "1": "1", "2": "2", "3": "3", "4": "4",
}


@register
class CompetitionCNN(MethodPlugin):
    """Wraps ABC1 KTC2023 competition entry (Beraldo et al., UFABC). CNN post-processor for EIT segmentation.

    The reconstruction runs as a subprocess to avoid TensorFlow / PyTorch
    dependency conflicts.  ABC1's main_python.py is called in its original
    directory (``KTC2023_Python_A01+/``) so relative file loads of
    ``Mesh_sparse.mat``, ``ultimate_cnn1.h5``, and ``TrainingData/ref.mat``
    all resolve correctly.

    The wrapper:
    1. Creates a temporary input directory containing a symlink (or copy on
       Windows) of the relevant ``data{N}.mat`` file from the framework's
       dataset root.
    2. Creates a temporary output directory.
    3. Calls: ``python main_python.py <input_dir> <output_dir> <category>``
    4. Reads the output ``.mat`` file, extracts the ``'reconstruction'`` field,
       and returns a ``(256, 256)`` uint8 array with labels ``{0, 1, 2}``.
    5. Cleans up both temporary directories.

    On any subprocess failure the error is logged and a zero array is returned
    so the benchmark can continue.
    """

    def reconstruct(self, batch: DataBatch) -> np.ndarray:  # noqa: C901
        tmp_input: Optional[str] = None
        tmp_output: Optional[str] = None

        if _ABC1_CWD is None or _ABC1_MAIN is None:
            # Warning already printed at import time — just return zeros silently.
            return np.zeros((256, 256), dtype=np.uint8)

        if not _TF_AVAILABLE:
            # Warning already printed at import time — just return zeros silently.
            return np.zeros((256, 256), dtype=np.uint8)

        try:
            # ------------------------------------------------------------------
            # 1. Locate the source data file for this sample.
            # ------------------------------------------------------------------
            # We need to find the raw data .mat file that main_python.py should
            # process.  The framework's DataBatch does not carry the original
            # file path, so we derive it from dataset_root / level / sample_id.
            #
            # sample_id is e.g. "level3_B"  →  level=3, letter="B"  →  data2.mat
            sample_id: str = str(getattr(batch, "sample_id", ""))
            level: int = int(batch.level)

            # Extract the letter portion of sample_id (everything after "levelN_")
            letter = ""
            if "_" in sample_id:
                letter = sample_id.split("_", 1)[1].upper()
            file_num = _SAMPLE_MAP.get(letter, "1")

            # Walk up from known relative paths to find the evaluation data root.
            # The framework is typically run from KTC_WORK_HIS/ and the config
            # specifies dataset_root: EvaluationData, so data files live at:
            #   EvaluationData/evaluation_datasets/level{N}/data{num}.mat
            # We search a prioritised list of candidate roots.
            framework_root = Path(__file__).resolve().parents[3]  # KTC_WORK_HIS/
            data_file = self._find_data_file(framework_root, level, file_num)

            if data_file is None:
                _logger.error(
                    "CompetitionCNN: cannot locate data file for "
                    "level=%d sample_id=%s — returning zeros.",
                    level,
                    sample_id,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            # ------------------------------------------------------------------
            # 1b. Output cache. The CNN is deterministic: the same input data
            # file + the same model weights always yield the same reconstruction,
            # and the input never changes between runs. So cache the output keyed
            # by (model fingerprint, data-file content hash, level) and skip the
            # ~22 s TensorFlow subprocess on repeat runs across processes.
            # Only successful results are cached (failures fall through to zeros).
            # ------------------------------------------------------------------
            cnn_key = None
            try:
                with open(data_file, "rb") as _df:
                    _data_fp = hashlib.md5(_df.read()).hexdigest()
                cnn_key = f"cnn|{_MODEL_FP}|{_data_fp}|L{int(level)}"
                _cached = _opcache.load(cnn_key)
                if _cached is not None:
                    return _cached
            except Exception:
                cnn_key = None  # caching is best-effort; never block the run

            # ------------------------------------------------------------------
            # 2. Build a temporary input directory with just the data .mat file.
            #
            # main_python.py lists *.mat files in input_path and filters out
            # files whose names start with "ref".  We copy (not symlink — more
            # reliable on Windows) the single data file so ABC1 processes
            # exactly one measurement.
            # ------------------------------------------------------------------
            tmp_input = tempfile.mkdtemp(prefix="abc1_in_")
            dest_name = data_file.name  # preserve original name, e.g. data1.mat
            shutil.copy2(str(data_file), os.path.join(tmp_input, dest_name))

            # ------------------------------------------------------------------
            # 3. Create a temporary output directory.
            # ------------------------------------------------------------------
            tmp_output = tempfile.mkdtemp(prefix="abc1_out_")

            # ------------------------------------------------------------------
            # 4. Run ABC1's main_python.py as a subprocess.
            # ------------------------------------------------------------------
            cmd = [
                _ABC1_PYTHON,
                _ABC1_MAIN,
                tmp_input,
                tmp_output,
                str(level),
            ]

            _logger.info(
                "CompetitionCNN: running subprocess — level=%d sample=%s data=%s",
                level,
                sample_id,
                data_file,
            )

            result = subprocess.run(
                cmd,
                cwd=_ABC1_CWD,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute hard limit per sample
            )

            if result.returncode != 0:
                _logger.error(
                    "CompetitionCNN: subprocess exited with code %d.\n"
                    "  stdout: %s\n  stderr: %s",
                    result.returncode,
                    result.stdout[-2000:] if result.stdout else "",
                    result.stderr[-2000:] if result.stderr else "",
                )
                return np.zeros((256, 256), dtype=np.uint8)

            # ------------------------------------------------------------------
            # 5. Read the output .mat file and extract 'reconstruction'.
            # ------------------------------------------------------------------
            import scipy.io  # imported here to keep the top-level import list clean

            output_files = [
                f for f in os.listdir(tmp_output) if f.endswith(".mat")
            ]
            if not output_files:
                _logger.error(
                    "CompetitionCNN: subprocess succeeded but no .mat file "
                    "found in output dir '%s'.",
                    tmp_output,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            out_path = os.path.join(tmp_output, output_files[0])
            mat = scipy.io.loadmat(out_path, squeeze_me=True, struct_as_record=False)

            if "reconstruction" not in mat:
                _logger.error(
                    "CompetitionCNN: output .mat '%s' has no 'reconstruction' "
                    "key.  Keys present: %s",
                    out_path,
                    [k for k in mat if not k.startswith("_")],
                )
                return np.zeros((256, 256), dtype=np.uint8)

            reconstruction = np.asarray(mat["reconstruction"], dtype=np.uint8)

            if reconstruction.shape != (256, 256):
                _logger.error(
                    "CompetitionCNN: reconstruction shape %s != (256, 256).",
                    reconstruction.shape,
                )
                return np.zeros((256, 256), dtype=np.uint8)

            self.validate_output(reconstruction)
            if cnn_key is not None:
                _opcache.save(cnn_key, reconstruction)  # cache only successful output
            return reconstruction

        except subprocess.TimeoutExpired:
            _logger.error(
                "CompetitionCNN: subprocess timed out for level=%d sample=%s.",
                int(batch.level),
                str(getattr(batch, "sample_id", "?")),
            )
            return np.zeros((256, 256), dtype=np.uint8)

        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "CompetitionCNN: unexpected error — %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return np.zeros((256, 256), dtype=np.uint8)

        finally:
            # ------------------------------------------------------------------
            # 6. Clean up temporary directories regardless of outcome.
            # ------------------------------------------------------------------
            for tmp_dir in (tmp_input, tmp_output):
                if tmp_dir and os.path.isdir(tmp_dir):
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:  # noqa: BLE001
                        pass  # best-effort; do not mask the real error

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _find_data_file(framework_root: Path, level: int, file_num: str) -> Optional[Path]:
        """Locate the raw voltage .mat file for a given level and file number.

        Searches the following candidate layouts in order:

        1. ``<framework_root>/EvaluationData/evaluation_datasets/level{N}/data{num}.mat``
        2. ``<framework_root>/EvaluationData/level{N}/data{num}.mat``
        3. ``<framework_root>/evaluation_datasets/level{N}/data{num}.mat``
        4. Current working directory variants of the above.
        """
        filename = f"data{file_num}.mat"
        cwd = Path.cwd()

        candidate_roots = [
            framework_root,
            cwd,
        ]
        candidate_subdirs = [
            Path("EvaluationData") / "evaluation_datasets" / f"level{level}",
            Path("EvaluationData") / f"level{level}",
            Path("evaluation_datasets") / f"level{level}",
        ]

        for root in candidate_roots:
            for subdir in candidate_subdirs:
                candidate = root / subdir / filename
                if candidate.exists():
                    return candidate

        return None
