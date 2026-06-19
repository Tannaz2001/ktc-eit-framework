"""Best-effort on-disk cache for expensive, run-invariant reconstruction operators.

The Jacobian / FEM forward-solve outputs depend only on the mesh, electrode
patterns, difficulty level, and reference voltages — never on the per-sample
measurements — so they are identical on every benchmark run.  Persisting them to
disk lets the expensive build happen once and be reused across *processes*,
including each fresh dashboard "Run" subprocess (which otherwise starts with an
empty in-memory cache and rebuilds everything).

Design notes:
  * Keyed by a STABLE string (never process-specific ``id()``), so the same key
    is produced on every run.
  * Fully best-effort: any load/save failure silently falls back to in-process
    computation, so results are never affected — this only changes speed.
  * Stored under ``outputs/.opcache/``.  Delete that folder to clear the cache
    (e.g. after changing the mesh or a method's defaults).
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

_DIR = Path("outputs") / ".opcache"


def _path_for(key: str) -> Path:
    return _DIR / (hashlib.md5(key.encode("utf-8")).hexdigest() + ".pkl")


def load(key: str):
    """Return the cached value for ``key`` or ``None`` (on miss or any error)."""
    p = _path_for(key)
    try:
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def save(key: str, value) -> None:
    """Persist ``value`` under ``key``.  Best-effort: failures are swallowed."""
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        p = _path_for(key)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)  # atomic — avoids half-written cache files
    except Exception:
        pass
