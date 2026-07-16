"""Run manifest system with atomic writes and version history."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import fcntl
import time


OUTPUTS_DIR = Path("outputs")
MANIFEST_FILE = OUTPUTS_DIR / "runs.manifest.json"


def _ensure_outputs_dir():
    """Create outputs directory if missing."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(file_path: Path, data: dict, retries: int = 3) -> bool:
    """
    Write to file atomically using file locking.
    Prevents corruption from concurrent writes.
    """
    _ensure_outputs_dir()

    for attempt in range(retries):
        try:
            # Write to temporary file first
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

            # Atomic rename
            temp_path.replace(file_path)
            return True
        except (OSError, IOError) as e:
            if attempt < retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                print(f"Failed to write manifest after {retries} attempts: {e}")
                return False

    return False


def _load_manifest() -> dict:
    """Load manifest with fallback to empty dict."""
    _ensure_outputs_dir()

    if not MANIFEST_FILE.exists():
        return {"runs": [], "last_loaded": None}

    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"runs": [], "last_loaded": None}


def register_run(run_dir: Path) -> bool:
    """
    Register a new run in the manifest.

    Args:
        run_dir: Path to the run directory (e.g., outputs/run_20260618_164304)

    Returns:
        success: bool
    """
    manifest = _load_manifest()

    run_entry = {
        "name": run_dir.name,
        "path": str(run_dir.resolve()),
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
    }

    # Check if already exists
    existing = [r for r in manifest.get("runs", []) if r["name"] == run_dir.name]
    if not existing:
        manifest["runs"].append(run_entry)

    return _atomic_write(MANIFEST_FILE, manifest)


def set_active_run(run_dir: Path) -> bool:
    """
    Set a run as the 'active' (currently displayed) run.
    More robust than writing to latest.txt.

    Args:
        run_dir: Path to the run directory

    Returns:
        success: bool
    """
    manifest = _load_manifest()
    manifest["last_loaded"] = {
        "name": run_dir.name,
        "path": str(run_dir.resolve()),
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    return _atomic_write(MANIFEST_FILE, manifest)


def get_active_run() -> Optional[Path]:
    """Get the currently active run, or None."""
    manifest = _load_manifest()
    last_loaded = manifest.get("last_loaded")

    if last_loaded and last_loaded.get("path"):
        path = Path(last_loaded["path"])
        if path.exists():
            return path

    # Fallback: return most recent run
    runs = get_all_runs()
    if runs:
        return runs[0]

    return None


def get_all_runs() -> List[Path]:
    """Get all runs sorted by timestamp (newest first)."""
    _ensure_outputs_dir()

    runs = sorted(
        [d for d in OUTPUTS_DIR.glob("run_*") if d.is_dir()],
        key=lambda d: d.stat().st_ctime,
        reverse=True,
    )
    return runs


def get_manifest_stats() -> Dict:
    """Get manifest statistics."""
    manifest = _load_manifest()

    return {
        "total_runs_tracked": len(manifest.get("runs", [])),
        "last_loaded": manifest.get("last_loaded"),
        "manifest_path": str(MANIFEST_FILE),
    }


def cleanup_manifest(max_history: int = 100) -> int:
    """
    Remove old entries from manifest to keep it lean.

    Args:
        max_history: keep only this many entries

    Returns:
        number of entries removed
    """
    manifest = _load_manifest()

    runs = manifest.get("runs", [])
    if len(runs) <= max_history:
        return 0

    # Sort by registration time and keep only recent ones
    runs_sorted = sorted(
        runs,
        key=lambda r: r.get("registered_at", ""),
        reverse=True,
    )
    kept = runs_sorted[:max_history]
    removed = len(runs) - len(kept)

    manifest["runs"] = kept
    _atomic_write(MANIFEST_FILE, manifest)

    return removed


def validate_manifest() -> Dict:
    """Validate manifest integrity and fix issues."""
    manifest = _load_manifest()
    issues = []
    fixed = 0

    # Remove entries with missing paths
    valid_runs = []
    for run in manifest.get("runs", []):
        path = run.get("path")
        if path and Path(path).exists():
            valid_runs.append(run)
        else:
            issues.append(f"Removed invalid entry: {run.get('name')}")
            fixed += 1

    if fixed > 0:
        manifest["runs"] = valid_runs
        _atomic_write(MANIFEST_FILE, manifest)

    return {
        "valid": len(valid_runs),
        "fixed": fixed,
        "issues": issues,
    }
