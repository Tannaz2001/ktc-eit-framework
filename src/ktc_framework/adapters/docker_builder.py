"""Build Docker method images from user-uploaded bundles and register them.

Bundle zip layout (all three files required):
    my_method.zip
    ├── algorithm.py      implements reconstruct(batch) -> ndarray
    ├── requirements.txt  pip dependencies
    └── ktc_config.yml    optional — name, base_image

Public API:
    build_method_from_bundle(zip_path)  ->  registered method name (str)

DockerBuildError is raised (with docker build stdout/stderr attached) if the
build fails so the caller can surface the exact pip / layer error to the user.

Subprocess persistence note:
After a successful build this module also writes a thin docker_runner.py into
external_methods/<name>/ so the benchmark subprocess (which re-scans that
directory on every run via load_cli_scripts) can re-discover and re-register
the Docker method without any extra framework changes.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ktc_framework.methods.method_plugin import MethodPlugin
from ktc_framework.registry import register_method

_logger = logging.getLogger(__name__)

_BRIDGE_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "methods" / "_template" / "bridge.py"
)
_DEFAULT_BASE_IMAGE = "python:3.10-slim"
_DOCKER_TIMEOUT = 600  # seconds — generous for heavy ML dependency installs


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DockerBuildError(Exception):
    """Raised when the docker build subprocess fails."""


# ---------------------------------------------------------------------------
# Serialization helpers (DataBatch → JSON for bridge.py)
# ---------------------------------------------------------------------------

def _encode_array(arr: Any) -> Any:
    if arr is None:
        return None
    a = np.asarray(arr)
    return {
        "data": base64.b64encode(a.tobytes()).decode("ascii"),
        "dtype": str(a.dtype),
        "shape": list(a.shape),
    }


def _encode_mesh(obj: Any) -> Any:
    """Best-effort recursive serialization of the mesh dict.

    numpy arrays are base64-encoded; scipy mat_struct fields are walked via
    __dict__; anything else that can't be serialized is silently omitted.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return _encode_array(obj)
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            encoded = _encode_mesh(v)
            if encoded is not None:
                out[str(k)] = encoded
        return out or None
    if isinstance(obj, (list, tuple)):
        return [_encode_mesh(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # scipy mat_struct and other objects: try walking their attrs
    try:
        if hasattr(obj, "__dict__"):
            return _encode_mesh(
                {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            )
    except Exception:
        pass
    return None


def serialize_batch(batch) -> str:
    """Serialize a DataBatch to a JSON string readable by bridge.py."""
    payload = {
        "voltages": _encode_array(batch.voltages),
        "injection_patterns": _encode_array(batch.injection_patterns),
        "ground_truth": _encode_array(batch.ground_truth),
        "level": int(batch.level),
        "sample_id": str(batch.sample_id),
        "mesh": _encode_mesh(getattr(batch, "mesh", None)),
        "reference_voltages": _encode_array(getattr(batch, "reference_voltages", None)),
        "measurement_patterns": _encode_array(
            getattr(batch, "measurement_patterns", None)
        ),
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Docker wrapper class factory
# ---------------------------------------------------------------------------

def _create_docker_wrapper_class(name: str, image_tag: str) -> type:
    """Return a MethodPlugin subclass that delegates reconstruct() to docker run."""

    class _DockerWrapper(MethodPlugin):
        def reconstruct(self, batch) -> np.ndarray:
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"ktc_docker_{name}_"))
            input_path = tmp_dir / "input.json"
            output_path = tmp_dir / "output.npy"
            try:
                input_path.write_text(serialize_batch(batch), encoding="utf-8")

                # Bind-mount the temp dir as /data inside the container.
                # On Windows + Docker Desktop the path is forwarded automatically.
                mount = f"{str(tmp_dir).replace(chr(92), '/')}:/data"
                cmd = [
                    "docker", "run", "--rm",
                    "-v", mount,
                    image_tag,
                    "/data/input.json",
                    "/data/output.npy",
                ]
                _logger.info("%s: docker run — level=%s sample=%s",
                             name, batch.level, batch.sample_id)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_DOCKER_TIMEOUT,
                )
                if result.returncode != 0:
                    _logger.error(
                        "%s: docker run exited %d\n  stdout: %s\n  stderr: %s",
                        name, result.returncode,
                        (result.stdout or "")[-2000:],
                        (result.stderr or "")[-2000:],
                    )
                    return np.zeros((256, 256), dtype=np.uint8)

                if not output_path.exists():
                    _logger.error("%s: output.npy not written by container", name)
                    return np.zeros((256, 256), dtype=np.uint8)

                arr = np.load(str(output_path)).astype(np.uint8)
                if arr.shape != (256, 256):
                    _logger.error("%s: unexpected output shape %s", name, arr.shape)
                    return np.zeros((256, 256), dtype=np.uint8)

                return arr

            except subprocess.TimeoutExpired:
                _logger.error("%s: docker run timed out (%ds)", name, _DOCKER_TIMEOUT)
                return np.zeros((256, 256), dtype=np.uint8)
            except Exception as exc:
                _logger.error("%s: unexpected error — %s", name, exc, exc_info=True)
                return np.zeros((256, 256), dtype=np.uint8)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    _DockerWrapper.__name__ = name
    _DockerWrapper.__qualname__ = name
    _DockerWrapper.__docker_image__ = image_tag
    return _DockerWrapper


# ---------------------------------------------------------------------------
# Persistence shim for benchmark subprocesses
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = '''\
"""Auto-generated KTC CLI shim — delegates to Docker image {image_tag}.

Discoverable by load_cli_scripts() (main() + argparse + __main__ guard).
Do not edit; regenerated by build_method_from_bundle().
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFolder")
    parser.add_argument("outputFolder")
    parser.add_argument("categoryNbr")
    args = parser.parse_args()

    level = int(args.categoryNbr)
    input_dir = args.inputFolder
    output_dir = args.outputFolder

    # Locate the single data*.mat + ref.mat in input_dir
    data_files = sorted(
        f for f in os.listdir(input_dir) if f.startswith("data") and f.endswith(".mat")
    )
    if not data_files:
        sys.exit(f"No data*.mat in {{input_dir}}")
    data_file = os.path.join(input_dir, data_files[0])
    ref_file = os.path.join(input_dir, "ref.mat")

    mat = scipy.io.loadmat(data_file, squeeze_me=True, struct_as_record=False)
    ref = scipy.io.loadmat(ref_file, squeeze_me=True, struct_as_record=False) if os.path.exists(ref_file) else {{}}

    voltages = np.asarray(mat.get("Uel", mat.get("voltages", np.zeros(76))), dtype="float32").flatten()
    Mpat = ref.get("Mpat") if ref else None
    injection_patterns = np.asarray(Mpat if Mpat is not None else np.zeros((32, 76)), dtype="float32")
    ground_truth = np.zeros((256, 256), dtype="uint8")
    sample_id = os.path.splitext(data_files[0])[0]

    def encode(arr):
        a = np.asarray(arr)
        return {{"data": base64.b64encode(a.tobytes()).decode(), "dtype": str(a.dtype), "shape": list(a.shape)}}

    payload = json.dumps({{
        "voltages": encode(voltages),
        "injection_patterns": encode(injection_patterns),
        "ground_truth": encode(ground_truth),
        "level": level,
        "sample_id": sample_id,
        "mesh": None,
        "reference_voltages": None,
        "measurement_patterns": None,
    }})

    tmp = tempfile.mkdtemp(prefix="ktc_dockershim_")
    try:
        input_json = os.path.join(tmp, "input.json")
        output_npy = os.path.join(tmp, "output.npy")
        with open(input_json, "w") as f:
            f.write(payload)

        mount = tmp.replace("\\\\", "/") + ":/data"
        result = subprocess.run(
            ["docker", "run", "--rm", "-v", mount,
             "{image_tag}", "/data/input.json", "/data/output.npy"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            sys.exit(f"docker run failed:\\n{{result.stderr[-2000:]}}")

        arr = np.load(output_npy).astype(np.uint8)
        os.makedirs(output_dir, exist_ok=True)
        scipy.io.savemat(os.path.join(output_dir, "1.mat"), {{"reconstruction": arr}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
'''


def _write_subprocess_shim(shim_path: Path, image_tag: str) -> None:
    """Write a KTC CLI-contract shim at shim_path so benchmark subprocesses
    can re-discover this Docker method.

    The shim must live at external_methods/{name}.py (root level) because
    load_cli_scripts() only scans path.glob("*.py") — subdirectory .py files
    are invisible to it.  The filename stem must equal the registered method
    name so derive_cli_method_name(stem) reproduces the exact same name that
    build_method_from_bundle() registered in-process.
    """
    shim_path.write_text(
        _RUNNER_TEMPLATE.format(image_tag=image_tag), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Bundle extraction + config parsing
# ---------------------------------------------------------------------------

_REQUIRED_FILES = ("algorithm.py", "requirements.txt")


def _safe_name(raw: str) -> str:
    """Convert an arbitrary string to a valid Python identifier."""
    name = re.sub(r"\W+", "_", raw).strip("_") or "DockerMethod"
    if not name[0].isalpha():
        name = f"Method_{name}"
    return name


def _extract_zip(zip_path: Path, dest: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise DockerBuildError(f"Not a valid zip file: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Zip-slip guard
            target = (dest / member.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise DockerBuildError(
                    f"Unsafe path in zip: {member.filename}"
                )
        zf.extractall(dest)

    # Flatten single top-level directory (GitHub-style zip layout)
    entries = [e for e in dest.iterdir() if not e.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(dest / item.name))
        inner.rmdir()


def _parse_config(work_dir: Path, zip_stem: str) -> tuple[str, str]:
    """Return (method_name, base_image) from ktc_config.yml (or safe defaults)."""
    cfg_path = work_dir / "ktc_config.yml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise DockerBuildError(f"Invalid ktc_config.yml: {exc}") from exc
        name = cfg.get("name") or zip_stem
        base_image = cfg.get("base_image") or _DEFAULT_BASE_IMAGE
    else:
        name = zip_stem
        base_image = _DEFAULT_BASE_IMAGE

    return _safe_name(str(name)), str(base_image)


def _write_dockerfile(work_dir: Path, base_image: str) -> None:
    dockerfile = (
        f"FROM {base_image}\n"
        "WORKDIR /app\n"
        "# numpy is always required by bridge.py\n"
        "RUN pip install --no-cache-dir numpy\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY algorithm.py .\n"
        "COPY bridge.py .\n"
        'ENTRYPOINT ["python", "bridge.py"]\n'
    )
    (work_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")


def _docker_build(work_dir: Path, image_tag: str) -> None:
    if not shutil.which("docker"):
        raise DockerBuildError(
            "docker executable not found. Install Docker Desktop and ensure "
            "it is running before uploading a Method Bundle."
        )

    result = subprocess.run(
        ["docker", "build", "-t", image_tag, "."],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=_DOCKER_TIMEOUT,
    )
    if result.returncode != 0:
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        raise DockerBuildError(
            f"docker build failed (exit {result.returncode}):\n\n"
            + output[-3000:]
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_external_image(name: str, image_tag: str, author: str = "") -> None:
    """Register a pre-built Docker image as a KTC method without building.

    The image is NOT pulled or validated — the caller must ensure
    ``docker pull <image_tag>`` succeeds before benchmarking.

    Actions:
    1. Writes a registry entry to ``configs/registered_methods.json``
       via ``method_registry_manager.add_method`` with ``status="active"``.
       If *author* is provided it is stored as an extra field in that entry.
    2. Registers an in-process wrapper so the current Streamlit session can
       run the method immediately.
    3. Writes a CLI-contract shim to ``external_methods/{name}.py`` so every
       fresh benchmark subprocess discovers the method via ``load_cli_scripts()``.

    Args:
        name:      Method name — must be a valid Python identifier.
        image_tag: Docker image reference, e.g. ``"docker.io/user/repo:latest"``.
        author:    Optional owner/author string stored in the registry entry.

    Raises:
        ValueError: If *name* is not a valid Python identifier.
    """
    from ktc_framework.method_registry_manager import add_method, load_registry, save_registry

    if not name.isidentifier():
        raise ValueError(
            f"Method name must be a valid Python identifier, got {name!r}. "
            "Use only letters, digits, and underscores; must not start with a digit."
        )

    add_method(name, image_tag, base_image="", status="active")

    if author:
        registry = load_registry()
        registry["methods"][name]["author"] = author
        save_registry(registry)

    wrapper_cls = _create_docker_wrapper_class(name, image_tag)
    register_method(wrapper_cls)

    Path("external_methods").mkdir(exist_ok=True)
    shim_path = Path("external_methods") / f"{name}.py"
    _write_subprocess_shim(shim_path, image_tag)
    _logger.info("link_external_image('%s'): linked %s, shim written", name, image_tag)


def remove_external_method(name: str) -> None:
    """Remove a registered Docker method from the JSON registry and delete its shim.

    Two actions are taken in order:

    1. ``delete_method(name)`` is called on ``method_registry_manager`` to
       remove the entry from ``configs/registered_methods.json``.
    2. The execution shim at ``external_methods/{name}.py`` is physically
       deleted so benchmark subprocesses no longer discover it via
       ``load_cli_scripts()``.

    Neither step raises if the entry or file is already absent — both are
    idempotent.  The in-process ``_METHODS`` registry entry is NOT cleared
    here; callers should unregister via ``ktc_framework.registry.unregister_method``
    and update ``session_state`` as needed.

    Args:
        name: Registered method name (exact key used in ``add_method``).
    """
    from ktc_framework.method_registry_manager import delete_method

    delete_method(name)
    shim_path = Path("external_methods") / f"{name}.py"
    shim_path.unlink(missing_ok=True)
    _logger.info("remove_external_method('%s'): registry entry deleted, shim removed", name)


def parse_bundle_name(zip_path: str) -> tuple[str, str]:
    """Extract and parse a bundle zip just enough to return (name, base_image).

    Performs full zip-slip validation and required-file checks so the caller
    gets the same errors as ``build_method_from_bundle``, but returns
    immediately without touching Docker.  The temp work directory is always
    cleaned up.

    Args:
        zip_path: Path to the .zip bundle.

    Returns:
        ``(name, base_image)`` derived from ``ktc_config.yml`` (or safe defaults).

    Raises:
        DockerBuildError: If the zip is invalid or missing required files.
    """
    zip_path = Path(zip_path)
    nonce = uuid.uuid4().hex[:8]
    work_dir = Path(tempfile.gettempdir()) / "ktc_parse" / nonce
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        _extract_zip(zip_path, work_dir)
        missing = [f for f in _REQUIRED_FILES if not (work_dir / f).exists()]
        if missing:
            raise DockerBuildError(
                f"Bundle is missing required file(s): {', '.join(missing)}. "
                "Expected layout:\n"
                "  my_method.zip\n"
                "  ├── algorithm.py     ← required\n"
                "  ├── requirements.txt ← required\n"
                "  └── ktc_config.yml   ← optional (name, base_image)\n"
            )
        return _parse_config(work_dir, zip_path.stem)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def build_method_async(bundle_path: Path, name: str, image_tag: str) -> None:
    """Background thread target: run docker build and update the JSON registry.

    Wraps ``build_method_from_bundle`` in a try/except and writes the final
    status (``"active"`` or ``"error"``) to ``configs/registered_methods.json``
    so the UI can reflect the true outcome on the next Streamlit rerender.

    ``bundle_path`` (the tmp zip) is deleted after the build attempt regardless
    of outcome — the caller must NOT delete it before the thread finishes.

    Args:
        bundle_path: Path to the saved .zip file (kept alive by caller).
        name:        Pre-registered method name (already in registry as
                     ``"building"``).
        image_tag:   Docker image tag, e.g. ``"ktc-mymethod:latest"``.
    """
    from ktc_framework.method_registry_manager import set_status

    try:
        build_method_from_bundle(str(bundle_path))
        set_status(name, "active")
        _logger.info("build_method_async('%s'): build complete, status=active", name)
        try:
            from dashboard.benchmark import append_method_to_config as _append_to_cfg
            _append_to_cfg(name)
        except Exception as _cfg_exc:
            _logger.warning(
                "build_method_async('%s'): could not append to config — %s", name, _cfg_exc
            )
    except Exception as exc:
        error_msg = str(exc)
        set_status(name, "error", error_msg)
        _logger.error("build_method_async('%s'): build failed — %s", name, error_msg)
    finally:
        bundle_path.unlink(missing_ok=True)


def build_method_from_bundle(zip_path: str) -> str:
    """Extract, build, and register a Docker method bundle.

    Args:
        zip_path: Path to the .zip containing algorithm.py, requirements.txt,
                  and optionally ktc_config.yml.

    Returns:
        The registered method name (usable in benchmark configs).

    Raises:
        DockerBuildError: if validation, docker build, or docker run pre-check fails.
    """
    zip_path = Path(zip_path)
    nonce = uuid.uuid4().hex[:8]
    work_dir = Path(tempfile.gettempdir()) / "ktc_uploads" / nonce
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Extract
        _extract_zip(zip_path, work_dir)

        # 2. Validate required files
        missing = [f for f in _REQUIRED_FILES if not (work_dir / f).exists()]
        if missing:
            raise DockerBuildError(
                f"Bundle is missing required file(s): {', '.join(missing)}. "
                "Expected layout:\n"
                "  my_method.zip\n"
                "  ├── algorithm.py     ← required\n"
                "  ├── requirements.txt ← required\n"
                "  └── ktc_config.yml   ← optional (name, base_image)\n"
            )

        # 3. Parse config
        name, base_image = _parse_config(work_dir, zip_path.stem)

        # 4. Copy bridge template into build context
        if not _BRIDGE_TEMPLATE.exists():
            raise DockerBuildError(
                f"Bridge template not found at {_BRIDGE_TEMPLATE}. "
                "Ensure src/ktc_framework/methods/_template/bridge.py exists."
            )
        shutil.copy2(_BRIDGE_TEMPLATE, work_dir / "bridge.py")

        # 5. Write Dockerfile
        _write_dockerfile(work_dir, base_image)

        # 6. Build image
        image_tag = f"ktc-{name.lower()}:latest"
        _docker_build(work_dir, image_tag)

        # 7. Register in-process wrapper
        wrapper_cls = _create_docker_wrapper_class(name, image_tag)
        register_method(wrapper_cls)

        # 8. Write shim to external_methods/{name}.py (ROOT level) so
        # load_cli_scripts()'s path.glob("*.py") picks it up in every fresh
        # benchmark subprocess.  The stem must equal `name` so that
        # derive_cli_method_name(stem) reproduces the exact registered name.
        Path("external_methods").mkdir(exist_ok=True)
        shim_path = Path("external_methods") / f"{name}.py"
        _write_subprocess_shim(shim_path, image_tag)

        _logger.info("Built and registered Docker method '%s' as %s", name, image_tag)
        return name

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
