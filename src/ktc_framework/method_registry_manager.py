"""JSON-based registry manager for Docker method images.

Persists method metadata to ``configs/registered_methods.json`` alongside the
existing ``configs/ktc_all_methods.yaml``.  Every write is wrapped in a
``FileLock`` so concurrent Streamlit reruns and background build subprocesses
never corrupt the file.

Schema
------
.. code-block:: json

    {
        "methods": {
            "<MethodName>": {
                "name":          "MethodName",
                "image_tag":     "ktc-methodname:latest",
                "status":        "active",
                "base_image":    "python:3.10-slim",
                "added_at":      "2026-01-01T00:00:00.000000",
                "updated_at":    "2026-01-01T00:00:00.000000",
                "published_url": null,
                "error":         null
            }
        }
    }

Valid ``status`` values
-----------------------
``"active"``     Method image built; ready to run.
``"inactive"``   Manually disabled; excluded from ``list_active_methods()``.
``"building"``   docker build in progress.
``"error"``      Last build or push failed; ``error`` field holds the message.
``"published"``  Pushed to Docker Hub; ``published_url`` is set.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (resolved to absolute so the module works from any working directory)
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_REGISTRY_PATH: Path = _REPO_ROOT / "configs" / "registered_methods.json"
_LOCK_PATH: Path = Path(str(_REGISTRY_PATH) + ".lock")
_LOCK_TIMEOUT: int = 5  # seconds

_EMPTY_REGISTRY: dict = {"methods": {}}

_VALID_STATUSES = frozenset({"active", "inactive", "building", "error", "published"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds").replace("+00:00", "")


def _atomic_write(registry: dict) -> None:
    """Write registry to disk inside a FileLock (timeout=5 s)."""
    try:
        with FileLock(_LOCK_PATH, timeout=_LOCK_TIMEOUT):
            _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _REGISTRY_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
            tmp.replace(_REGISTRY_PATH)
    except Timeout:
        raise RuntimeError(
            f"Could not acquire lock on {_REGISTRY_PATH} within {_LOCK_TIMEOUT}s. "
            "Another process may be writing the registry."
        )


def _validate_status(status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. Must be one of: {sorted(_VALID_STATUSES)}"
        )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def load_registry() -> dict:
    """Read and return the full registry dict.

    If ``configs/registered_methods.json`` does not yet exist the empty
    skeleton ``{"methods": {}}`` is both written to disk and returned.
    """
    if not _REGISTRY_PATH.exists():
        _atomic_write(_EMPTY_REGISTRY)
        return {"methods": {}}

    try:
        raw = _REGISTRY_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        _logger.warning("Registry file unreadable (%s); reinitialising.", exc)
        _atomic_write(_EMPTY_REGISTRY)
        return {"methods": {}}

    if not isinstance(data, dict) or "methods" not in data:
        _logger.warning("Registry file malformed; reinitialising.")
        _atomic_write(_EMPTY_REGISTRY)
        return {"methods": {}}

    return data


def save_registry(registry: dict) -> None:
    """Atomically persist *registry* to disk.

    The write is protected by ``filelock.FileLock`` with a 5-second timeout
    so concurrent Streamlit reruns cannot produce a partial write.

    Args:
        registry: Full registry dict (must contain a ``"methods"`` key).

    Raises:
        ValueError: If *registry* does not contain a ``"methods"`` key.
        RuntimeError: If the lock cannot be acquired within the timeout.
    """
    if "methods" not in registry:
        raise ValueError("registry dict must contain a 'methods' key")
    _atomic_write(registry)


def add_method(
    name: str,
    image_tag: str,
    base_image: str = "python:3.10-slim",
    status: str = "active",
) -> dict:
    """Add or overwrite a method entry in the registry.

    If a method with the same *name* already exists it is fully replaced
    (``added_at`` is preserved from the existing entry on overwrite).

    Args:
        name:       Registered method name (must be a valid Python identifier).
        image_tag:  Local Docker image tag, e.g. ``"ktc-mymethod:latest"``.
        base_image: Base image used in the Dockerfile, e.g. ``"python:3.10-slim"``.
        status:     Initial status (default ``"active"``).

    Returns:
        The newly written entry dict.

    Raises:
        ValueError: If *name* is not a valid Python identifier or *status* is invalid.
        RuntimeError: If the lock cannot be acquired.
    """
    if not name.isidentifier():
        raise ValueError(
            f"Method name must be a valid Python identifier, got {name!r}"
        )
    _validate_status(status)

    registry = load_registry()
    now = _now()
    existing = registry["methods"].get(name, {})

    entry: dict = {
        "name": name,
        "image_tag": image_tag,
        "status": status,
        "base_image": base_image,
        "added_at": existing.get("added_at", now),
        "updated_at": now,
        "published_url": existing.get("published_url"),
        "error": None,
    }
    registry["methods"][name] = entry
    save_registry(registry)
    _logger.info("Registry: added/updated method '%s' (status=%s)", name, status)
    return entry


def set_status(
    name: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update the ``status`` (and optionally ``error``) of an existing entry.

    Creates a minimal entry if *name* is not yet in the registry (e.g. when
    a build subprocess updates status before ``add_method`` is called by the
    parent process).

    Args:
        name:   Method name.
        status: New status string.
        error:  Error message to store when status is ``"error"``; cleared
                automatically for all other statuses unless explicitly set.

    Raises:
        ValueError: If *status* is not a recognised value.
        RuntimeError: If the lock cannot be acquired.
    """
    _validate_status(status)

    registry = load_registry()
    now = _now()
    entry = registry["methods"].setdefault(name, {
        "name": name,
        "image_tag": "",
        "status": status,
        "base_image": "",
        "added_at": now,
        "updated_at": now,
        "published_url": None,
        "error": None,
    })
    entry["status"] = status
    entry["updated_at"] = now
    entry["error"] = error if status == "error" else None
    registry["methods"][name] = entry
    save_registry(registry)
    _logger.info("Registry: set status '%s' -> '%s'", name, status)


def delete_method(name: str) -> bool:
    """Remove a method entry from the registry.

    Args:
        name: Method name to remove.

    Returns:
        ``True`` if the entry existed and was removed, ``False`` otherwise.

    Raises:
        RuntimeError: If the lock cannot be acquired.
    """
    registry = load_registry()
    if name not in registry["methods"]:
        _logger.debug("Registry: delete_method('%s') — not found, no-op", name)
        return False
    del registry["methods"][name]
    save_registry(registry)
    _logger.info("Registry: deleted method '%s'", name)
    return True


def list_active_methods() -> list[str]:
    """Return names of all methods whose status is ``"active"`` or ``"published"``.

    Inactive, building, and error entries are excluded.

    Returns:
        Sorted list of method name strings.
    """
    registry = load_registry()
    return sorted(
        name
        for name, entry in registry["methods"].items()
        if entry.get("status") in {"active", "published"}
    )


# ---------------------------------------------------------------------------
# Docker Hub publish
# ---------------------------------------------------------------------------


def publish_method(name: str) -> str:
    """Tag and push a built Docker image to Docker Hub.

    Reads credentials from the environment:

    - ``DOCKER_USERNAME`` — Docker Hub account name (used for both login and
      the target image namespace).
    - ``DOCKER_PASSWORD`` — Docker Hub access token or password, passed to
      ``docker login`` via stdin (never as a CLI argument) so it does not
      appear in process listings or shell history.

    Workflow
    --------
    1. Look up *name* in the registry to get its local ``image_tag``.
    2. ``docker login --username <usr> --password-stdin``
    3. ``docker tag <image_tag> <usr>/ktc-<name.lower()>:latest``
    4. ``docker push <usr>/ktc-<name.lower()>:latest``
    5. Update ``published_url`` and set ``status = "published"`` in the
       registry.
    6. Return the fully-qualified Docker Hub URL.

    Args:
        name: Registered method name (must already exist in the registry with
              a valid ``image_tag``).

    Returns:
        The published image URL, e.g.
        ``"docker.io/myuser/ktc-mymethod:latest"``.

    Raises:
        KeyError:   If *name* is not in the registry.
        ValueError: If ``image_tag`` is empty or DOCKER_USERNAME / DOCKER_PASSWORD
                    are not set.
        RuntimeError: If any docker CLI step fails.
    """
    registry = load_registry()
    if name not in registry["methods"]:
        raise KeyError(
            f"Method '{name}' not found in registry. "
            "Call add_method() after a successful docker build first."
        )

    entry = registry["methods"][name]
    image_tag = entry.get("image_tag", "").strip()
    if not image_tag:
        raise ValueError(
            f"Registry entry for '{name}' has an empty image_tag. "
            "Re-run build_method_from_bundle() to rebuild the image."
        )

    username = os.environ.get("DOCKER_USERNAME", "").strip()
    password = os.environ.get("DOCKER_PASSWORD", "").strip()
    if not username:
        raise ValueError(
            "DOCKER_USERNAME environment variable is not set. "
            "Export it before calling publish_method()."
        )
    if not password:
        raise ValueError(
            "DOCKER_PASSWORD environment variable is not set. "
            "Export it (or set it as a Docker Hub access token) before calling publish_method()."
        )

    remote_tag = f"{username}/ktc-{name.lower()}:latest"
    published_url = f"docker.io/{remote_tag}"

    # Step 1 — login (password via stdin, never as CLI argument)
    _logger.info("publish_method('%s'): docker login as '%s'", name, username)
    login_result = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin"],
        input=password,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if login_result.returncode != 0:
        raise RuntimeError(
            f"docker login failed (exit {login_result.returncode}):\n"
            + (login_result.stderr or login_result.stdout or "")[-1000:]
        )

    # Step 2 — tag
    _logger.info("publish_method('%s'): tagging %s -> %s", name, image_tag, remote_tag)
    tag_result = subprocess.run(
        ["docker", "tag", image_tag, remote_tag],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if tag_result.returncode != 0:
        raise RuntimeError(
            f"docker tag failed (exit {tag_result.returncode}):\n"
            + (tag_result.stderr or tag_result.stdout or "")[-1000:]
        )

    # Step 3 — push
    _logger.info("publish_method('%s'): pushing %s", name, remote_tag)
    push_result = subprocess.run(
        ["docker", "push", remote_tag],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if push_result.returncode != 0:
        set_status(name, "error", error=push_result.stderr[-500:] or push_result.stdout[-500:])
        raise RuntimeError(
            f"docker push failed (exit {push_result.returncode}):\n"
            + (push_result.stderr or push_result.stdout or "")[-2000:]
        )

    # Step 4 — update registry
    entry["published_url"] = published_url
    entry["status"] = "published"
    entry["updated_at"] = _now()
    entry["error"] = None
    registry["methods"][name] = entry
    save_registry(registry)

    _logger.info("publish_method('%s'): published -> %s", name, published_url)
    return published_url
