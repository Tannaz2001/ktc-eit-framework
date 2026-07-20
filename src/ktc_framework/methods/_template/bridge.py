"""Bridge between the KTC host benchmarker and a Docker-isolated algorithm.

Invoked inside the container by DockerMethodWrapper as:
    python bridge.py <input.json> <output.npy>

The host serializes each DataBatch field to a JSON-safe envelope:
    numpy arrays  -> {"data": "<base64>", "dtype": "float32", "shape": [...]},
    scalars/strs  -> JSON native types
    None fields   -> null

This file has zero ktc_framework dependency so it runs in any Python 3.10+
environment where only numpy is installed.
"""
from __future__ import annotations

import base64
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Minimal DataBatch stub
# Field names must exactly match ktc_framework.types.DataBatch so that
# algorithm.py can use batch.voltages, batch.level, etc. unchanged.
# ---------------------------------------------------------------------------

class DataBatch(NamedTuple):
    voltages: np.ndarray
    injection_patterns: np.ndarray
    ground_truth: np.ndarray
    level: int
    sample_id: str
    mesh: Any = None
    reference_voltages: Any = None
    measurement_patterns: Any = None


# ---------------------------------------------------------------------------
# ktc_framework stubs
# Injected into sys.modules BEFORE importing algorithm.py so that common
# framework imports resolve without the full package installed:
#   from ktc_framework.methods.method_plugin import MethodPlugin
#   from ktc_framework.registry import register_method
#   from ktc_framework.types import DataBatch
# ---------------------------------------------------------------------------

class _MethodPlugin:
    """Lightweight stand-in for MethodPlugin; reconstruct() is all that matters."""

    def reconstruct(self, batch: DataBatch) -> np.ndarray:  # type: ignore[return]
        raise NotImplementedError

    def validate_output(self, output: np.ndarray) -> None:
        arr = np.asarray(output)
        if arr.shape != (256, 256):
            raise ValueError(f"Output must be (256, 256), got {arr.shape}")
        if not np.all(np.isin(arr, [0, 1, 2])):
            raise ValueError("Output labels must be in {0, 1, 2}")


def _register_noop(cls):
    """Stand-in for @register_method — does nothing inside the container."""
    return cls


def _inject_stubs() -> None:
    def _mod(name: str, **attrs: Any) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    stubs = {
        "ktc_framework": _mod("ktc_framework"),
        "ktc_framework.methods": _mod("ktc_framework.methods"),
        "ktc_framework.methods.method_plugin": _mod(
            "ktc_framework.methods.method_plugin",
            MethodPlugin=_MethodPlugin,
        ),
        "ktc_framework.registry": _mod(
            "ktc_framework.registry",
            register_method=_register_noop,
            register=_register_noop,
        ),
        "ktc_framework.types": _mod(
            "ktc_framework.types",
            DataBatch=DataBatch,
        ),
    }
    for name, mod in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

def _decode_array(obj: Any) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict) and "data" in obj and "shape" in obj:
        raw = base64.b64decode(obj["data"])
        return np.frombuffer(raw, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
    return np.asarray(obj)


def _decode_mesh(obj: Any) -> Any:
    """Recursively decode mesh dict, converting any array blobs."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "data" in obj and "shape" in obj:
            return _decode_array(obj)
        return {k: _decode_mesh(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_mesh(v) for v in obj]
    return obj


def _load_batch(input_path: str) -> DataBatch:
    with open(input_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    return DataBatch(
        voltages=_decode_array(raw["voltages"]),
        injection_patterns=_decode_array(raw["injection_patterns"]),
        ground_truth=_decode_array(raw["ground_truth"]),
        level=int(raw["level"]),
        sample_id=str(raw["sample_id"]),
        mesh=_decode_mesh(raw.get("mesh")),
        reference_voltages=_decode_array(raw.get("reference_voltages")),
        measurement_patterns=_decode_array(raw.get("measurement_patterns")),
    )


# ---------------------------------------------------------------------------
# Algorithm import + reconstruct discovery
# ---------------------------------------------------------------------------

def _import_algorithm(path: str = "/app/algorithm.py"):
    spec = importlib.util.spec_from_file_location("algorithm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load algorithm from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_reconstruct(module: types.ModuleType):
    """Return a bound reconstruct(batch) callable from the imported module.

    Search order mirrors the host framework's priority:
    1. First class defined in algorithm.py that has a reconstruct method.
    2. Module-level reconstruct(batch) function.
    """
    for _, obj in inspect.getmembers(module, inspect.isclass):
        # Skip re-exported stubs (e.g. _MethodPlugin injected via sys.modules)
        if getattr(obj, "__module__", "") != "algorithm":
            continue
        if callable(getattr(obj, "reconstruct", None)):
            try:
                return obj().reconstruct
            except TypeError:
                continue  # constructor requires args

    if callable(getattr(module, "reconstruct", None)):
        return module.reconstruct

    raise AttributeError(
        "No reconstruct(batch) callable found in algorithm.py. "
        "Define a class with def reconstruct(self, batch) or a module-level "
        "def reconstruct(batch)."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <input.json> <output.npy>\n")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    _inject_stubs()

    try:
        batch = _load_batch(input_path)
    except Exception as exc:
        sys.stderr.write(f"bridge: failed to deserialize DataBatch — {exc}\n")
        sys.exit(1)

    try:
        module = _import_algorithm()
    except Exception as exc:
        sys.stderr.write(f"bridge: failed to import algorithm.py — {exc}\n")
        sys.exit(1)

    try:
        reconstruct = _find_reconstruct(module)
    except AttributeError as exc:
        sys.stderr.write(f"bridge: {exc}\n")
        sys.exit(1)

    try:
        result = reconstruct(batch)
    except Exception as exc:
        sys.stderr.write(f"bridge: reconstruct(batch) raised — {exc}\n")
        sys.exit(1)

    arr = np.asarray(result, dtype=np.uint8)
    if arr.shape != (256, 256):
        sys.stderr.write(
            f"bridge: expected output shape (256, 256), got {arr.shape}\n"
        )
        sys.exit(1)

    np.save(output_path, arr)


if __name__ == "__main__":
    main()
