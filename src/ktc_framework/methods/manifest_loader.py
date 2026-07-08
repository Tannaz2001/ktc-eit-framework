"""Parse and validate method.yaml manifests for ML method bundles."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ManifestError(Exception):
    """Raised when a method.yaml is missing, malformed, or invalid."""


_DEFAULT_SAMPLE_MAP = {"A": "1", "B": "2", "C": "3", "1": "1", "2": "2", "3": "3", "4": "4"}
_DEFAULT_ARGS = ["input_dir", "output_dir", "level"]
_DEFAULT_TIMEOUT = 300
_DEFAULT_PYTHON_VERSIONS = ["3.12", "3.11", "3.10"]


@dataclass
class MethodManifest:
    name: str
    description: str
    python_versions: list[str]
    check_import: str | None
    env_override: str | None
    entry_point: str
    args_order: list[str]
    working_dir: str
    timeout: int
    weights: list[str]
    sample_map: dict[str, str]
    bundle_dir: Path = field(repr=False)


def load_manifest(yaml_path: Path) -> MethodManifest:
    """Load and validate a method.yaml file into a MethodManifest."""
    if not yaml_path.exists():
        raise ManifestError(f"Manifest not found: {yaml_path}")

    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"Invalid YAML in {yaml_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"Manifest must be a YAML mapping, got {type(raw).__name__}")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ManifestError("Manifest must have a 'name' field (string)")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ManifestError(f"Method name must be a valid Python identifier, got '{name}'")

    runtime: dict[str, Any] = raw.get("runtime") or {}
    solver: dict[str, Any] = raw.get("solver") or {}

    entry_point = solver.get("entry_point")
    if not entry_point or not isinstance(entry_point, str):
        raise ManifestError("Manifest must have solver.entry_point (string)")

    bundle_dir = yaml_path.parent

    manifest = MethodManifest(
        name=name,
        description=str(raw.get("description", "")),
        python_versions=_as_str_list(runtime.get("python_versions", _DEFAULT_PYTHON_VERSIONS)),
        check_import=runtime.get("check_import"),
        env_override=runtime.get("env_override"),
        entry_point=entry_point,
        args_order=_as_str_list(solver.get("args", _DEFAULT_ARGS)),
        working_dir=str(solver.get("working_dir", ".")),
        timeout=int(solver.get("timeout", _DEFAULT_TIMEOUT)),
        weights=_as_str_list(raw.get("weights", [])),
        sample_map=_merge_sample_map(raw.get("sample_map")),
        bundle_dir=bundle_dir,
    )

    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError(f"Invalid manifest for '{name}': " + "; ".join(errors))

    return manifest


def validate_manifest(manifest: MethodManifest) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []

    ep_path = manifest.bundle_dir / manifest.entry_point
    if not ep_path.exists():
        errors.append(f"entry_point '{manifest.entry_point}' not found in {manifest.bundle_dir}")

    if manifest.timeout < 1:
        errors.append(f"timeout must be >= 1, got {manifest.timeout}")

    for w in manifest.weights:
        wp = manifest.bundle_dir / w
        if not wp.exists():
            errors.append(f"weights file '{w}' not found in {manifest.bundle_dir}")

    cwd = manifest.bundle_dir / manifest.working_dir
    if not cwd.is_dir():
        errors.append(f"working_dir '{manifest.working_dir}' is not a directory")

    return errors


def extract_bundle(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a method bundle .zip into dest_dir.

    Handles two layouts:
      - method.yaml at the zip root → extract directly to dest_dir
      - method.yaml one level deep (single root dir) → flatten into dest_dir

    Returns the directory containing method.yaml.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ManifestError(f"Not a valid zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ManifestError("Empty zip file")

        if "method.yaml" in names:
            dest_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(dest_dir)
            return dest_dir

        top_dirs = {n.split("/")[0] for n in names if "/" in n}
        if len(top_dirs) == 1:
            root_prefix = top_dirs.pop() + "/"
            if root_prefix + "method.yaml" in names:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for member in zf.infolist():
                    if member.filename.startswith(root_prefix) and member.filename != root_prefix:
                        member.filename = member.filename[len(root_prefix):]
                        zf.extract(member, dest_dir)
                return dest_dir

        raise ManifestError(
            "Zip must contain method.yaml at root or inside a single top-level directory"
        )


def extract_archive(zip_path: Path, dest_dir: Path) -> Path:
    """Extract any method repository zip into dest_dir.

    Unlike extract_bundle, this is used for raw ML/KTC repository archives that
    do not yet contain method.yaml. If the archive has one top-level directory
    (the normal GitHub zip layout), that directory is flattened so the generated
    method.yaml can live directly in dest_dir.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ManifestError(f"Not a valid zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ManifestError("Empty zip file")

        dest_dir.mkdir(parents=True, exist_ok=True)
        top_dirs = {n.split("/")[0] for n in names if "/" in n}
        has_root_files = any("/" not in n.rstrip("/") for n in names if not n.endswith("/"))
        root_prefix = f"{next(iter(top_dirs))}/" if len(top_dirs) == 1 and not has_root_files else ""

        for member in zf.infolist():
            if member.is_dir():
                continue
            target_name = member.filename
            if root_prefix:
                if not target_name.startswith(root_prefix):
                    continue
                target_name = target_name[len(root_prefix):]
            if not target_name:
                continue

            target = (dest_dir / target_name).resolve()
            if dest_dir.resolve() not in target.parents and target != dest_dir.resolve():
                raise ManifestError(f"Unsafe zip member path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as dst:
                dst.write(src.read())

    return dest_dir


def _as_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [str(v) for v in val]
    return [str(val)]


def _merge_sample_map(user_map: Any) -> dict[str, str]:
    merged = dict(_DEFAULT_SAMPLE_MAP)
    if isinstance(user_map, dict):
        for k, v in user_map.items():
            merged[str(k)] = str(v)
    return merged
